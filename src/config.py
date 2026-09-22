"""Central configuration for the joke-RL project.

Every other module imports its settings from here, so there is exactly one
place to change a model name, a reward weight, or an output path. Values are
grouped into frozen dataclasses by pipeline stage (data -> SFT -> reward -> RL)
so that related knobs stay together and nothing can be mutated by accident at
run time.

Two things are resolved at import time from the environment:

  * PROFILE  -- "local" (tiny model, fast, runs on a MacBook) or "colab"
                (real model, real training). Auto-detected, overridable.
  * PATHS    -- every directory can be redirected with an env var, so the
                Gradio app can load checkpoints straight from Google Drive.

Usage:
    from src.config import PATHS, SFT, REWARD, RL, get_device, PROFILE
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------
# Profile: local (tiny model) vs colab (real model)
# --------------------------------------------------------------------------
# The single switch that decides whether we are smoke-testing the code or
# actually training. Everything else keys off this, so a script never needs to
# know where it is running.
#
#   auto (default)  -> "colab" when running inside Google Colab, else "local"
#   JOKE_RL_PROFILE=colab   force the real model (e.g. a rented GPU box)
#   JOKE_RL_PROFILE=local   force the tiny model (e.g. debugging on Colab)

VALID_PROFILES = ("local", "colab")


def _detect_profile() -> str:
    """Resolve the profile without importing torch.

    We deliberately do not use `torch.cuda.is_available()` here: config is
    imported by scripts that never touch a model, and importing torch costs
    seconds. Colab announces itself through both a module and an env var.
    """
    override = os.environ.get("JOKE_RL_PROFILE", "").strip().lower()
    if override:
        if override not in VALID_PROFILES:
            raise ValueError(
                f"JOKE_RL_PROFILE={override!r} is not valid; "
                f"expected one of {VALID_PROFILES}"
            )
        return override

    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    return "colab" if in_colab else "local"


PROFILE = _detect_profile()
IS_COLAB = PROFILE == "colab"
IS_LOCAL = PROFILE == "local"


def _env_path(var: str) -> Path | None:
    """Read a path from the environment, expanding ~ and $VARS."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    return Path(os.path.expandvars(raw)).expanduser()


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is resolved relative to this file, so scripts work no matter
# which directory you launch them from (local shell, Colab, or a notebook).

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent.parent

    @property
    def data(self) -> Path:
        """Dataset root. Override with JOKE_RL_DATA_DIR.

        On Colab, point this at Drive so a disconnected runtime does not cost
        you the cleaned dataset:
            os.environ["JOKE_RL_DATA_DIR"] = "/content/drive/MyDrive/joke-rl/data"
        """
        return _env_path("JOKE_RL_DATA_DIR") or (self.root / "data")

    @property
    def raw(self) -> Path:
        """Untouched downloads (Short Jokes CSV, r/Jokes dump)."""
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        """Cleaned, deduplicated, context-tagged jokes ready for training."""
        return self.data / "processed"

    @property
    def models(self) -> Path:
        """Checkpoint root. Override with JOKE_RL_MODELS_DIR.

        This is the one that matters most. Colab runtimes are wiped when they
        disconnect, so training writes to Drive:
            os.environ["JOKE_RL_MODELS_DIR"] = "/content/drive/MyDrive/joke-rl/models"

        The Gradio app reads the same variable, so pointing it at a synced
        Drive folder locally is all it takes to demo a Colab-trained model.
        """
        return _env_path("JOKE_RL_MODELS_DIR") or (self.root / "models")

    @property
    def sft_model(self) -> Path:
        return self.models / "sft"

    @property
    def reward_model(self) -> Path:
        return self.models / "reward"

    @property
    def rl_model(self) -> Path:
        return self.models / "rl"

    @property
    def dpo_model(self) -> Path:
        return self.models / "dpo"

    @property
    def outputs(self) -> Path:
        """Plots, eval tables, reward curves."""
        return self.root / "outputs"

    @property
    def votes(self) -> Path:
        """Human preference pairs collected by the Gradio app."""
        return self.root / "app" / "votes"

    def ensure(self) -> None:
        """Create every directory that scripts write into."""
        for p in (self.raw, self.processed, self.models, self.outputs, self.votes):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        """Human-readable summary, for printing at the top of a script or
        notebook so it is obvious where output is going."""
        rows = [
            ("profile", PROFILE),
            ("root", self.root),
            ("data", self.data),
            ("models", self.models),
            ("outputs", self.outputs),
        ]
        return "\n".join(f"  {k:<10} {v}" for k, v in rows)


PATHS = Paths()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
# The model is trained on a flat prompt format so that at inference time we can
# hand it "<context> coffee <joke>" and let it complete the joke. Keeping the
# tokens here means the data script and the generation code can never disagree.

CONTEXT_TOKEN = "<context>"
JOKE_TOKEN = "<joke>"
END_TOKEN = "<|endoftext|>"


def format_prompt(context: str) -> str:
    """Build the exact string the model is conditioned on. Note the trailing
    space: GPT-2's tokenizer treats ' word' and 'word' as different tokens."""
    return f"{CONTEXT_TOKEN} {context.strip().lower()} {JOKE_TOKEN} "


def format_example(context: str, joke: str) -> str:
    """A full training line: prompt + target joke + end-of-text."""
    return f"{format_prompt(context)}{joke.strip()}{END_TOKEN}"


@dataclass(frozen=True)
class DataConfig:
    # Contexts the model is trained to condition on. Anything outside this list
    # still works at inference (the model sees it as free text) but is
    # out-of-distribution, so quality will drop.
    contexts: tuple[str, ...] = (
        "coffee",
        "university exams",
        "programming",
        "work",
        "food",
        "animals",
        "relationships",
        "technology",
        "sports",
        "money",
        "school",
        "travel",
        "health",
        "music",
        "general",
    )
    # Jokes outside this word range are dropped: too short is usually a
    # fragment, too long usually a story that a small model cannot learn.
    min_words: int = 4
    max_words: int = 60
    max_chars: int = 400
    # Near-duplicate removal threshold (TF-IDF cosine similarity).
    dedup_threshold: float = 0.90
    # Minimum reddit score for an r/Jokes row to count as "funny" when
    # building the humour classifier's positive class.
    reddit_min_score: int = 100
    # Fraction of the cleaned corpus held out for evaluation.
    test_size: float = 0.05
    seed: int = 42


DATA = DataConfig()


# --------------------------------------------------------------------------
# Stage 1 — Supervised fine-tuning (SFT) with LoRA
# --------------------------------------------------------------------------
# Two models, chosen by PROFILE:
#
#   local -> sshleifer/tiny-gpt2   2 layers, ~100K params, downloads in seconds.
#            Its output is gibberish and always will be -- it is randomly
#            initialised. That is fine: locally we are testing that the *code*
#            runs, not that the model is good.
#
#   colab -> gpt2                  124M params, the real thing.
#
# Both share GPT-2's architecture and tokenizer, so a script that works on one
# works on the other with no edits. Switch with JOKE_RL_PROFILE, or per-run
# with JOKE_RL_BASE_MODEL=<any hf id>.

# LoRA injects adapters into the attention projections, and those are named
# differently per architecture. Getting this wrong is a silent failure: PEFT
# attaches to nothing, training runs, loss barely moves, and nothing warns you.
# So we resolve it from the model name rather than trusting a hardcoded value.
LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "gpt2": ("c_attn",),                                   # fused qkv, a Conv1D
    "gpt_neo": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "qwen": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "opt": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "pythia": ("query_key_value",),
}


def lora_targets_for(model_name: str) -> tuple[str, ...]:
    """Map a HF model id to its attention projection names.

    Matches on substring, so "sshleifer/tiny-gpt2" and "distilgpt2" both
    resolve to the gpt2 entry. Raises rather than guessing, because a wrong
    guess here fails silently at training time.
    """
    key = model_name.lower()
    for family, modules in LORA_TARGETS.items():
        if family in key:
            return modules
    raise ValueError(
        f"Unknown architecture for {model_name!r}: no LoRA target modules known.\n"
        f"Add an entry to LORA_TARGETS in src/config.py. Known families: "
        f"{', '.join(LORA_TARGETS)}.\n"
        f"To find the right names: "
        f"print([n for n, _ in model.named_modules()])"
    )


@dataclass(frozen=True)
class SFTConfig:
    # The two ends of the switch.
    tiny_model: str = "sshleifer/tiny-gpt2"
    full_model: str = "gpt2"          # upgrade path on Colab: "Qwen/Qwen2.5-0.5B"

    max_length: int = 128

    # LoRA: instead of updating all 124M weights, we train small rank-r adapter
    # matrices injected into the attention projections. ~0.5% of the parameters
    # are trainable, which is what makes this feasible without a big GPU.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 25
    save_steps: int = 500

    # Locally we only want to prove the training loop runs, so cap the dataset
    # at a size that finishes in seconds. None means "use everything".
    local_max_train_samples: int | None = 200
    local_epochs: int = 1

    @property
    def base_model(self) -> str:
        """The model this run actually uses.

        Precedence: explicit env var > profile default. The env var exists so
        you can test the real model locally (once it is downloaded) without
        pretending to be Colab.
        """
        return os.environ.get("JOKE_RL_BASE_MODEL", "").strip() or (
            self.full_model if IS_COLAB else self.tiny_model
        )

    @property
    def lora_target_modules(self) -> tuple[str, ...]:
        """Attention projections to adapt, derived from base_model.

        Note for GPT-2: c_attn is a Conv1D whose weight is stored transposed
        relative to nn.Linear. PEFT detects this and flips fan_in_fan_out
        itself, printing a UserWarning -- expected on GPT-2, absent on Qwen.
        """
        return lora_targets_for(self.base_model)

    @property
    def effective_epochs(self) -> int:
        return self.epochs if IS_COLAB else self.local_epochs

    @property
    def max_train_samples(self) -> int | None:
        return None if IS_COLAB else self.local_max_train_samples

    @property
    def is_tiny(self) -> bool:
        """True when the loaded model is the throwaway stand-in. Use this to
        skip quality assertions that a random model cannot possibly pass."""
        return self.base_model == self.tiny_model


SFT = SFTConfig()


# --------------------------------------------------------------------------
# Stage 2 — Reward model
# --------------------------------------------------------------------------
# The RL stage needs a single scalar per generated joke. We build it as a
# weighted sum of five signals rather than one learned score, because a single
# learned humour head is very easy for the policy to game (it will find one
# weird string that scores 0.99 and emit it forever). The relevance, novelty,
# toxicity and length terms exist specifically to close off those exploits.
#
# Weights should sum to roughly 1.0 for the total to stay interpretable.

@dataclass(frozen=True)
class RewardConfig:
    humor_model: str = "distilbert-base-uncased"       # we fine-tune this ourselves
    similarity_model: str = "all-MiniLM-L6-v2"          # sentence-transformers
    toxicity_model: str = "s-nlp/roberta_toxicity_classifier"

    w_humor: float = 0.50       # learned humour classifier probability
    w_relevance: float = 0.25   # cosine(context, joke) via sentence embeddings
    w_novelty: float = 0.15     # 1 - max similarity to any training joke
    w_toxicity: float = 0.10    # (1 - toxicity), so clean jokes score higher
    w_length: float = 0.00      # soft penalty, off by default; see length_* below

    # Length shaping: jokes inside [target_min, target_max] words get no
    # penalty, outside it the score falls off linearly.
    length_target_min: int = 8
    length_target_max: int = 40

    # Novelty is measured against the training corpus; a generation that is a
    # near-copy above this similarity is treated as fully unoriginal.
    novelty_threshold: float = 0.85

    # Humour classifier training
    humor_epochs: int = 2
    humor_batch_size: int = 16
    humor_learning_rate: float = 2e-5
    humor_max_length: int = 128

    @property
    def weights(self) -> dict[str, float]:
        return {
            "humor": self.w_humor,
            "relevance": self.w_relevance,
            "novelty": self.w_novelty,
            "toxicity": self.w_toxicity,
            "length": self.w_length,
        }


REWARD = RewardConfig()


# --------------------------------------------------------------------------
# Stage 3 — Reinforcement learning (GRPO)
# --------------------------------------------------------------------------
# GRPO is preferred over PPO here: it needs no separate value network, which
# halves memory use and removes a whole class of tuning headaches on a T4.
# It samples `num_generations` jokes per prompt and pushes the policy toward
# the above-average ones within each group.

@dataclass(frozen=True)
class RLConfig:
    algorithm: str = "grpo"           # "grpo" or "ppo"
    num_generations: int = 8          # samples per prompt (the "group" in GRPO)
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 1e-5       # much lower than SFT: RL is unstable
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.95

    # KL penalty against the frozen SFT model. This is the main guard against
    # reward hacking: it keeps the policy from drifting into degenerate text
    # that happens to score well. Raise it if generations become repetitive.
    kl_beta: float = 0.05

    total_steps: int = 500
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42


RL = RLConfig()


# --------------------------------------------------------------------------
# Stage 4 — Human feedback / DPO
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.1                 # DPO's implicit-KL strength
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 5e-6
    min_votes_to_train: int = 100     # don't bother training on fewer pairs
    votes_file: str = "preferences.jsonl"


DPO = DPOConfig()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    # Contexts held out for the SFT vs RL vs RL+HF comparison.
    eval_contexts: tuple[str, ...] = (
        "coffee",
        "university exams",
        "programming",
        "food",
        "work",
    )
    samples_per_context: int = 20
    distinct_n: tuple[int, ...] = (1, 2, 3)   # diversity metric orders
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 64
    seed: int = 42


EVAL = EvalConfig()


# --------------------------------------------------------------------------
# Gradio app
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    title: str = "Joke Generator — which one is funnier?"
    server_port: int = 7860
    share: bool = False
    temperature: float = 0.95
    top_p: float = 0.95
    max_new_tokens: int = 64

    # Which checkpoint the demo serves. None means "best available", resolved
    # by model_io.resolve_checkpoint() as dpo -> rl -> sft -> untrained base.
    # Override with JOKE_RL_APP_MODEL for an exact path, or JOKE_RL_MODELS_DIR
    # to point the whole project at a Drive folder.
    model_stage: str | None = None

    # Fold LoRA weights into the base model at load time: faster generation,
    # and the app never trains, so there is no downside here.
    merge_adapter: bool = True


APP = AppConfig()


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------

def get_device() -> str:
    """Pick the best available backend.

    Order is CUDA (Colab T4) -> MPS (Apple Silicon) -> CPU, so the identical
    script runs in both places with no edits. Import torch lazily so that
    importing this config stays cheap for scripts that never touch a model.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype():
    """Half precision only on CUDA.

    fp16 on MPS is still flaky for training (silent NaNs in some ops), and it
    is meaningless on CPU, so both fall back to float32.
    """
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.float32


def use_fp16() -> bool:
    """Whether to pass fp16=True to a HF Trainer. CUDA only."""
    import torch

    return torch.cuda.is_available()
