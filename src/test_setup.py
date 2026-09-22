"""Environment smoke test — run this first, and any time something breaks.

It answers one question: can this machine actually train and run the models
this project needs? It checks the interpreter, the libraries, the compute
backend, and then does a real end-to-end load-and-generate with GPT-2 plus a
LoRA wrap, because "the import worked" and "the model runs" are different
claims.

Which model it tests follows the profile in config.py:
    local (default on this Mac)  -> SFT.tiny_model, downloads in seconds
    colab                        -> SFT.full_model, the real thing

Run:  python -m src.test_setup            # whatever the profile says
      python -m src.test_setup --full     # force the real model
      python -m src.test_setup --model distilgpt2
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

from src.config import PATHS, PROFILE, SFT, format_prompt, get_device, get_dtype

# Minimum versions the rest of the pipeline assumes.
REQUIRED = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "sentence_transformers",
    "pandas",
    "sklearn",
    "gradio",
]

PASS = "  ok  "
FAIL = " FAIL "


def _line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<28} {detail}")


def check_python() -> bool:
    ok = sys.version_info >= (3, 9)
    _line(PASS if ok else FAIL, "python", f"{platform.python_version()} on {platform.machine()}")
    return ok


def check_packages() -> bool:
    """Import each dependency and report its version.

    We import rather than read a requirements file, because a package can be
    listed and still fail to load (wrong architecture wheel, broken install).
    """
    import importlib

    all_ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            _line(PASS, name, version)
        except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
            _line(FAIL, name, f"import failed: {exc}")
            all_ok = False
    return all_ok


def check_device() -> bool:
    """Report the compute backend and prove a tensor op actually runs on it.

    On Apple Silicon `mps.is_available()` can be True while a specific op still
    falls over, so we do a real matmul rather than trusting the flag.
    """
    import torch

    device = get_device()
    _line(PASS, "torch.cuda.is_available", str(torch.cuda.is_available()))
    _line(PASS, "torch.backends.mps", f"built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")

    try:
        x = torch.randn(64, 64, device=device)
        y = (x @ x.T).sum().item()
        _line(PASS, f"matmul on {device}", f"finite={y == y}")  # NaN != NaN
        return True
    except Exception as exc:  # noqa: BLE001
        _line(FAIL, f"matmul on {device}", str(exc))
        return False


def check_generation(model_name: str) -> bool:
    """Load the model, wrap it in LoRA, and generate a joke-shaped continuation.

    This is the real test: it exercises the tokenizer, the special-token setup,
    the device placement, PEFT's adapter injection, and sampling — the same path
    the SFT stage will use.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    dtype = get_dtype()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    _line(PASS, f"load {model_name}", f"{time.time() - t0:.1f}s, {model.num_parameters()/1e6:.1f}M params")

    # GPT-2 ships without a padding token. Batched training and generation both
    # need one, and reusing EOS is the standard fix (the attention mask keeps
    # the padding from affecting the output).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        _line(PASS, "pad_token", f"unset -> reusing eos ({tokenizer.eos_token!r})")

    # Wrap in LoRA exactly as the SFT stage will, to confirm the target module
    # names in config.py match this architecture.
    # Resolve the target modules from the model actually being tested, which
    # is the whole point of lora_targets_for(): --model qwen would otherwise
    # attach adapters to GPT-2 module names and silently train nothing.
    from src.config import lora_targets_for

    targets = lora_targets_for(model_name)
    lora = LoraConfig(
        r=SFT.lora_r,
        lora_alpha=SFT.lora_alpha,
        lora_dropout=SFT.lora_dropout,
        target_modules=list(targets),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    _line(PASS, "LoRA wrap", f"{targets} — {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    peft_model.to(device)
    peft_model.eval()

    prompt = format_prompt("coffee")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = peft_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    _line(PASS, "generate", f"{out.shape[1] - inputs['input_ids'].shape[1]} new tokens in {time.time() - t0:.1f}s")

    print()
    print("  prompt :", repr(prompt))
    print("  output :", text.strip())
    print()
    if model_name == SFT.tiny_model:
        print("  (Output is gibberish by design: this is a randomly-initialised")
        print("   miniature model. It proves the code path works, nothing more.)")
    else:
        print("  (Output is nonsense right now — this is the base model with an")
        print("   untrained LoRA adapter. It proves the plumbing works, not the model.)")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="joke-rl environment check")
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"force the real model ({SFT.full_model}) instead of the profile default",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help=f"force the tiny model ({SFT.tiny_model})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="explicit model id to test with (overrides --tiny and config)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model:
        model_name = args.model
    elif args.full:
        model_name = SFT.full_model
    elif args.tiny:
        model_name = SFT.tiny_model
    else:
        model_name = SFT.base_model

    print("=" * 72)
    print(f"joke-rl environment check  [profile: {PROFILE}]")
    print("=" * 72)

    PATHS.ensure()
    _line(PASS, "project dirs", f"created under {PATHS.root}")
    _line(PASS, "models dir", str(PATHS.models))

    results = {
        "python": check_python(),
        "packages": check_packages(),
        "device": check_device(),
    }
    print("-" * 72)

    if not all(results.values()):
        print("\nEnvironment checks failed — skipping the model test.")
        return 1

    try:
        results["generation"] = check_generation(model_name)
    except Exception as exc:  # noqa: BLE001
        _line(FAIL, "generation", str(exc))
        results["generation"] = False

    print("=" * 72)
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All checks passed. Device: {get_device()}, dtype: {get_dtype()}, model: {model_name}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
