"""Loading and saving trained models.

The Gradio app, the evaluation scripts and the RL loop all need to load a
checkpoint that may be a LoRA adapter, a full model, or nothing at all (fall
back to the untrained base). They should not each reimplement that logic, so it
lives here.

The path is configurable at every level, which is what makes a Colab-trained
model usable locally:

    JOKE_RL_MODELS_DIR=/content/drive/MyDrive/joke-rl/models   # base directory
    JOKE_RL_APP_MODEL=/path/to/a/specific/checkpoint           # exact override

Usage:
    from src.model_io import load_policy, save_policy, resolve_checkpoint

    model, tokenizer, info = load_policy()          # best available checkpoint
    model, tokenizer, info = load_policy("rl")      # a specific stage
    model, tokenizer, info = load_policy(some_path) # an explicit directory
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.config import PATHS, SFT, get_device, get_dtype

# Which checkpoint to prefer when the caller does not name one. Later stages
# are better models, so we search from the end of the pipeline backwards.
STAGE_PRIORITY = ("dpo", "rl", "sft")


def _stage_dir(stage: str) -> Path:
    return PATHS.models / stage


def _is_checkpoint(path: Path) -> bool:
    """A directory is loadable if it holds either a LoRA adapter or a full model."""
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() or (path / "config.json").exists()


def _is_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


@dataclass(frozen=True)
class LoadedModel:
    """What was actually loaded — worth printing in the app and in eval tables,
    so a result is never ambiguous about which checkpoint produced it."""

    path: Path | None
    stage: str
    base_model: str
    is_adapter: bool
    is_trained: bool

    def __str__(self) -> str:
        if not self.is_trained:
            return f"UNTRAINED base model ({self.base_model})"
        kind = "LoRA adapter" if self.is_adapter else "full model"
        return f"{self.stage} ({kind}) on {self.base_model} — {self.path}"


def available_checkpoints() -> dict[str, Path]:
    """Every loadable checkpoint under the models directory, by stage name."""
    if not PATHS.models.is_dir():
        return {}
    return {
        d.name: d
        for d in sorted(PATHS.models.iterdir())
        if _is_checkpoint(d)
    }


def resolve_checkpoint(stage: str | Path | None = None) -> tuple[Path | None, str]:
    """Decide which checkpoint to load.

    Precedence, highest first:
      1. an explicit path passed by the caller
      2. an explicit stage name passed by the caller ("sft", "rl", "dpo")
      3. JOKE_RL_APP_MODEL, an exact path from the environment
      4. the best available stage, searching dpo -> rl -> sft
      5. nothing — the caller falls back to the untrained base model

    Returns (path, stage_label). A None path means "use the base model".
    """
    # 1 / 2 — caller was explicit
    if stage is not None:
        candidate = Path(stage).expanduser()
        if candidate.is_dir() or os.sep in str(stage):
            if not _is_checkpoint(candidate):
                raise FileNotFoundError(
                    f"No model found at {candidate}.\n"
                    f"Expected an adapter_config.json or config.json inside it."
                )
            return candidate, candidate.name
        # a bare stage name
        path = _stage_dir(str(stage))
        if not _is_checkpoint(path):
            found = ", ".join(available_checkpoints()) or "none"
            raise FileNotFoundError(
                f"No trained '{stage}' model under {PATHS.models}.\n"
                f"Available checkpoints: {found}.\n"
                f"If the model lives on Google Drive, set JOKE_RL_MODELS_DIR."
            )
        return path, str(stage)

    # 3 — environment override
    env_override = os.environ.get("JOKE_RL_APP_MODEL", "").strip()
    if env_override:
        path = Path(os.path.expandvars(env_override)).expanduser()
        if not _is_checkpoint(path):
            raise FileNotFoundError(
                f"JOKE_RL_APP_MODEL points at {path}, which is not a model directory."
            )
        return path, path.name

    # 4 — best available
    found = available_checkpoints()
    for name in STAGE_PRIORITY:
        if name in found:
            return found[name], name

    # 5 — nothing trained yet
    return None, "base"


def _adapter_base_model(path: Path) -> str:
    """Read the base model an adapter was trained on.

    PEFT records this in adapter_config.json at save time, which is what lets a
    checkpoint be loaded without the caller remembering what it was built from.
    """
    cfg = json.loads((path / "adapter_config.json").read_text())
    recorded = cfg.get("base_model_name_or_path")
    if not recorded:
        return SFT.base_model
    return recorded


def load_policy(
    stage: str | Path | None = None,
    *,
    merge_adapter: bool = False,
    device: str | None = None,
) -> tuple[object, object, LoadedModel]:
    """Load a model + tokenizer, ready to generate.

    `merge_adapter` folds the LoRA weights into the base model. That makes
    generation faster and is what you want for the demo app, but it makes the
    model untrainable, so leave it off during training.

    Returns (model, tokenizer, info).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    dtype = get_dtype()
    path, stage_label = resolve_checkpoint(stage)

    if path is None:
        # Nothing trained yet. Loading the base model keeps the app runnable
        # before stage 2 exists, which is useful for building the UI early.
        base_name = SFT.base_model
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        info = LoadedModel(None, "base", base_name, False, False)

    elif _is_adapter(path):
        base_name = _adapter_base_model(path)
        tokenizer = AutoTokenizer.from_pretrained(path if (path / "tokenizer_config.json").exists() else base_name)
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        model = PeftModel.from_pretrained(base, path)
        if merge_adapter:
            model = model.merge_and_unload()
        info = LoadedModel(path, stage_label, base_name, True, True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        info = LoadedModel(path, stage_label, str(path), False, True)

    # GPT-2 ships without a pad token; batched generation needs one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer, info


def save_policy(model, tokenizer, stage: str) -> Path:
    """Save a checkpoint under the configured models directory.

    Because PATHS.models honours JOKE_RL_MODELS_DIR, calling this on Colab with
    that variable pointed at Drive means training output survives the runtime
    being recycled — which it will be.
    """
    path = _stage_dir(stage)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
