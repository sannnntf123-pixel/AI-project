# CLAUDE.md — project context for future sessions

> **Read [`PROGRESS.md`](PROGRESS.md) first** — it records what is done, what is
> half-finished, and the exact next step. This file is the stable context;
> PROGRESS.md is the moving state.

## What this is
A university AI course project: an AI that generates creative, funny jokes in a
**user-chosen context** (e.g. "coffee", "university exams", "programming"),
trained with **reinforcement learning**.

## Hard constraints (do not violate)
- **The models must be built and trained here.** Calling an external LLM API to
  generate the jokes is **not allowed** by the course. Writing code is fine.
- **Never commit large data or model files.** `data/` and `models/` are
  gitignored. The `.venv/` must never be tracked.
- The owner is a CS student who has to **understand and present** this project.
  Briefly explain what each piece of code does and why — the reasoning matters
  as much as the code.

## How to work on this
- **Step by step.** Finish and test one stage before starting the next.
- Keep code clean and modular. **All** config values (model names, reward
  weights, paths, hyperparameters) live in `src/config.py` — one source of truth.
- Training code must run on **both CUDA (Colab T4) and MPS/CPU (MacBook Air)**.
  Use `config.get_device()` / `get_dtype()` / `use_fp16()`; never hardcode
  `.cuda()` or `fp16=True`.
- Heavy training happens on **Google Colab**, so training stages need
  **Colab-ready notebooks** in `notebooks/`, not just scripts.

## Profiles and paths (important)

**Never hardcode a model name or a path.** Both are resolved in `src/config.py`.

`PROFILE` is `local` or `colab`, auto-detected (Colab is detected via
`google.colab` in `sys.modules` or `COLAB_RELEASE_TAG`), overridable with
`JOKE_RL_PROFILE`:

- `local` -> `SFT.tiny_model` = `sshleifer/tiny-gpt2`. ~100K params, random
  weights, gibberish output **by design**. Use it to test that code runs. Never
  assert on output quality under this profile — check `SFT.is_tiny`.
- `colab` -> `SFT.full_model` = `gpt2`. Real training.

Always read `SFT.base_model` (a property), never `tiny_model`/`full_model`
directly. Likewise use `SFT.lora_target_modules`, which calls
`lora_targets_for(base_model)` — hardcoding `("c_attn",)` breaks the moment the
base model changes, and it breaks *silently*.

Locally, `SFT.effective_epochs` is 1 and `SFT.max_train_samples` is 200, so a
training smoke test finishes in seconds.

Environment variables (all optional, documented in `.env.example`):

| Var | Effect |
|---|---|
| `JOKE_RL_PROFILE` | `local` / `colab` |
| `JOKE_RL_BASE_MODEL` | override the model id for one run |
| `JOKE_RL_MODELS_DIR` | move `PATHS.models` (e.g. to Drive) |
| `JOKE_RL_DATA_DIR` | move `PATHS.data` |
| `JOKE_RL_APP_MODEL` | serve one exact checkpoint in the app |

Load and save checkpoints through `src/model_io.py` only:
`load_policy(stage=None)` returns `(model, tokenizer, info)` and resolves
`dpo -> rl -> sft -> untrained base`; `save_policy(model, tokenizer, stage)`
writes under `PATHS.models`, which honours the Drive override.

## Environment
- MacBook Air, Apple Silicon (arm64), macOS. Python 3.11.9, venv at `.venv/`.
- MPS available; **no CUDA locally**.
- Key versions: torch 2.14, transformers 5.17, trl 1.13, peft 0.21,
  datasets 5.0, accelerate 1.15, sentence-transformers 6.1, gradio 6.28,
  pandas 3.0, scikit-learn 1.9.
- Note: transformers 5.x and trl 1.x are **recent majors** with API changes from
  the 4.x/0.x tutorials most docs are written against. Check signatures against
  the installed version rather than trusting from memory.
- Remote: https://github.com/sannnntf123-pixel/AI-project.git

## Architecture (6 stages)

| # | Stage | Where | Status |
|---|-------|-------|--------|
| 0 | Scaffolding, config, setup test | local | **done** |
| 1 | Data: download, clean, dedupe, filter, context-tag | local | next |
| 2 | SFT: LoRA fine-tune on context-labeled jokes | Colab | not started |
| 3 | Reward model: 5-signal weighted sum | Colab | not started |
| 4 | RL: GRPO with KL penalty vs. the SFT model | Colab | not started |
| 5 | Human feedback: Gradio A/B vote -> DPO | local | not started |
| 6 | Evaluation: SFT vs RL vs RL+HF, plots | both | not started |

### 1. Data
Sources: **Short Jokes** (Kaggle) and **r/Jokes** (has upvote scores).
Clean, deduplicate, filter offensive content, tag each joke with a context label.
Training format (defined in `config.py`, never hardcode it elsewhere):
```
<context> programming <joke> ...joke text...<|endoftext|>
```

### 2. SFT
LoRA (PEFT) fine-tune of a small pretrained model. Default `gpt2` (124M) for
speed; `Qwen/Qwen2.5-0.5B` is the upgrade path on Colab. Only `SFT.base_model`
and `SFT.lora_target_modules` need to change to switch.

### 3. Reward model — weighted sum of five signals
- **humor** (0.50) — a DistilBERT classifier *we train*, on r/Jokes upvotes or
  the ColBERT humor dataset
- **relevance** (0.25) — sentence-transformers cosine similarity(context, joke)
- **novelty** (0.15) — penalty for near-copies of training jokes
- **toxicity** (0.10) — penalty
- **length** (0.00, off by default) — soft penalty outside 8–40 words

It is a composite *on purpose*: a single learned humour head is trivially gamed
by the policy. The other four terms exist to close off those exploits.

### 4. RL
**GRPO** (preferred over PPO: no value network, so half the memory and far less
tuning on a T4) via HF TRL, with a **KL penalty against the frozen SFT model**.
**Watch for reward hacking** — this is a known failure mode here and a good
thing to document in the report. Log reward curves.

### 5. Human feedback
A **Gradio** app: user enters a context, sees two jokes, picks the funnier one.
Votes are saved as preference pairs (`app/votes/preferences.jsonl`) for **DPO**
or reward-model retraining.

### 6. Evaluation
Compare **SFT vs RL vs RL+human feedback** on: humor score, relevance, novelty,
diversity (**distinct-n**), and **blind human ratings**. Plot reward curves.

## Layout
```
data/        datasets            (gitignored)
models/      checkpoints         (gitignored)
notebooks/   Colab notebooks
src/         Python code
app/         Gradio demo
tests/       tests
outputs/     plots, eval tables  (gitignored)
```

## Conventions
- Run scripts as modules from the project root: `python -m src.test_setup`.
- `src/config.py` exports `PATHS, DATA, SFT, REWARD, RL, DPO, EVAL, APP` and the
  helpers `format_prompt()`, `format_example()`, `get_device()`, `get_dtype()`,
  `use_fp16()`.
- `PATHS.ensure()` creates every output directory; call it at the top of scripts
  that write files.
