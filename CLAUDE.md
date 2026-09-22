# CLAUDE.md — project context for future sessions

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
