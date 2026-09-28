# PROGRESS — where the project stands

Read this first when picking the project back up. Architecture and conventions
live in `CLAUDE.md`; this file tracks **what is done and what to do next**.

---

## Last session — 2026-09-28

### ✅ Stage 0 — Scaffolding (done)

| File | What it is |
|---|---|
| `src/config.py` | Single source of truth. Frozen dataclasses per stage: `PATHS, DATA, SFT, REWARD, RL, DPO, EVAL, APP`. Helpers: `format_prompt()`, `format_example()`, `get_device()`, `lora_targets_for()`. |
| `src/model_io.py` | `load_policy()` / `save_policy()` / `resolve_checkpoint()`. Handles LoRA adapters and full models, searches `dpo → rl → sft`, falls back to the untrained base. |
| `src/test_setup.py` | Environment smoke test. |
| `.env.example` | Every environment variable, documented. |

**Profiles:** `local` → `sshleifer/tiny-gpt2` (gibberish by design, seconds to
load); `colab` → `gpt2`. Auto-detected, override with `JOKE_RL_PROFILE`.
LoRA target modules are derived from the model id, never hardcoded.

**Paths:** `JOKE_RL_MODELS_DIR` / `JOKE_RL_DATA_DIR` redirect everything, so
Colab writes to Drive and the local Gradio app reads the same checkpoint.

### ✅ Task 0b — real GPT-2 downloaded (done)

Verified: 124.4M params, 589,824 LoRA params trainable (0.47%), generates on MPS.

Two things that cost time and will recur:
- A **176KB partial blob** left at the real cache path made `snapshot_download`
  report success in 0.00s **without downloading anything**. If a model loads
  with `Error while deserializing header: incomplete metadata`, delete the blob
  under `~/.cache/huggingface/hub/models--<name>/blobs/` and re-fetch.
- **HF's Xet backend 404s for anonymous requests.** Workaround:
  `HF_HUB_DISABLE_XET=1`. Use it for every large HF download on this machine.

### ✅ Stage 1 — Data (done)

`src/data_download.py` — both corpora, from Hugging Face not Kaggle (the
original Short Jokes needs a Kaggle token; `Fraser/short-jokes` is a loading
script, which `datasets` 5.x refuses to execute):

| Source | Rows | Scores |
|---|---|---|
| `ysharma/short_jokes` | 231,657 | no |
| `SocialGrep/one-million-reddit-jokes` | 1,000,000 | **yes** |

Both downloaded to `data/raw/` (25MB + 176MB). Reddit jokes are split across
`title` (setup) and `selftext` (punchline) and are joined on load.

`src/data_clean.py` — eight stages, cheap filters first:

```
normalise → tombstones → length → exact dedup → near-dedup
  → offensive → meta-posts → context tagging → split
```

Near-dedup uses **MinHash + LSH** (`datasketch`), not all-pairs TF-IDF: at 1.2M
rows that would be ~7×10¹¹ comparisons.

`tests/test_data_clean.py` — **44 tests, all passing.**

#### Results of the first full run

| Stage | Removed |
|---|---|
| raw | 1,231,657 |
| dead/empty (`[removed]`, `[deleted]`) | 421,779 |
| length (outside 4–60 words) | 87,981 |
| exact duplicates | 111,026 |
| near duplicates | 25,986 |
| offensive | 10,610 |
| **final** | **574,275 (46.6% kept)** |

447,361 rows carry upvote scores. Context split: `general` 42%, `animals` 10.5%,
`relationships` 8.9%, `food` 8.4%, then a long tail.

---

## ✅ Task 1c — cleaner re-run (done 2026-09-28)

`data/processed/` now reflects the fixed filters.

| Stage | Removed |
|---|---|
| raw | 1,231,657 |
| dead/empty | 421,779 |
| length | 87,981 |
| exact duplicates | 111,026 |
| near duplicates | 25,986 |
| offensive | 10,845 |
| reddit meta-posts | 5,564 |
| **final** | **568,476 (46.2% kept)** |

Train 540,052 / test 28,424. 442,189 rows carry upvote scores.

Both defects verified gone — 0 occurrences of British-spelling slurs,
digit-substituted slurs, and reddit meta-posts. 44 tests pass.

---

## ⚠️ OPEN TASK 1d — context labels are noisy (START HERE)

**This is the most important open issue, and it affects the core of the project.**

Context tagging is two-pass: keyword rules cover 29.5%, and sentence-embedding
similarity assigns the remaining 70.5%. The embedding pass is too permissive.

Proxy precision — how often a joke labelled with a context actually contains one
of that context's own keywords:

| Context | Keyword-confirmed |
|---|---|
| coffee | **22.6%** |
| food | 43.9% |
| programming | 44.3% |
| animals | 57.5% |
| university exams | 60.8% |
| relationships | 61.1% |

Spot-checking confirms it. Sampled "coffee" jokes include a bar joke and a
stolen-glasses joke with no coffee anywhere; "programming" includes a mosquito
-net charity joke (the keyword rule matched "program being started") and a
date-format pun.

**Why this matters:** if the context label is noise, SFT teaches the model to
*ignore* the context token. Then the RL relevance reward (0.25 weight) spends
its budget fighting the SFT model instead of improving jokes — and the headline
demo ("write me a joke about coffee") produces jokes unrelated to coffee.

### Options, cheapest first

1. **Raise `DATA.context_min_similarity`** from `0.25`. For normalised cosine
   against `"a joke about X"`, 0.25 is barely above noise. Try 0.35–0.45, and
   measure the precision table above at each value. Anything below the floor
   becomes `general`, which is honest.
2. **Keyword-only tagging** (`--no-embed`). High precision, but `general` grows
   from 42% to ~70% and the small contexts shrink to a few thousand rows each.
3. **Tighten the keyword rules.** `"program"` matching "program being started"
   is a word-boundary problem; `"ai "` and `"git "` are similarly loose.
4. **Accept the noise but downsample `general`** so the model at least sees
   balanced context conditioning.

Recommended: (1) + (3) together, then re-check the precision table.

---

## Two findings worth putting in the report


**High reddit score ≠ funny.** The three highest-scoring posts in the raw dump
are an obituary for the subreddit's founder, a net-neutrality protest, and a
broken-keyboard bit. r/Jokes upvotes community drama as readily as jokes. Any
naive "top-scored = funniest" labelling for the stage-3 humour classifier would
train on meta-commentary. This is why the meta-post filter runs *before* the
score is used.

**Score distribution is brutally skewed:** median 1, p90 36, p95 123, max
142,733. `DATA.reddit_min_score = 100` sits at roughly p95 and keeps 56,222
jokes (5.6%) as the positive class — a defensible cut, but it is a *choice* and
should be justified, not presented as obvious.

**Coverage warning for the demo:** `programming` (1.0%) and `university exams`
(1.0%) are the two smallest context buckets, and they are two of the three
contexts in the project pitch. The model will be weakest on exactly what gets
demoed. Options: broaden those keyword lists, oversample them during SFT, or
pick different demo contexts.

---

## ▶ NEXT — Stage 2: supervised fine-tuning

After task 1c. This is the first stage that needs a GPU, so it belongs in a
**Colab notebook**, not a local script.

### Steps
1. **`notebooks/02_sft.ipynb`** — Colab-ready. Opening cells:
   ```python
   !git clone https://github.com/sannnntf123-pixel/AI-project.git && cd AI-project
   !pip install -q -r requirements.txt
   from google.colab import drive; drive.mount('/content/drive')
   import os
   os.environ["JOKE_RL_MODELS_DIR"] = "/content/drive/MyDrive/joke-rl/models"
   os.environ["JOKE_RL_DATA_DIR"]   = "/content/drive/MyDrive/joke-rl/data"
   ```
   Upload `data/processed/*.jsonl` to Drive first — do **not** commit them.
2. **`src/train_sft.py`** — importable by the notebook so the logic is version
   controlled rather than living in notebook cells. Use TRL's `SFTTrainer`,
   `SFT.lora_target_modules`, `use_fp16()`, and `save_policy(model, tok, "sft")`.
3. **Smoke-test locally first**: `python -m src.train_sft` under the `local`
   profile runs 1 epoch on 200 samples with the tiny model in seconds. If that
   passes, the Colab run is very unlikely to fail on a code bug.
4. Train on Colab, confirm the checkpoint lands in Drive, then load it locally
   with `load_policy("sft")` and generate for a few contexts.

### Decisions still open
- `gpt2` or `Qwen/Qwen2.5-0.5B` for the real run. Qwen gives noticeably better
  jokes; only `SFT.full_model` needs to change (LoRA targets follow automatically).
- Whether to downsample the 42% `general` bucket so the model actually learns
  to condition on context rather than ignoring it.
- Whether to filter the SFT set by score (e.g. reddit score ≥ 10) — trains on
  better jokes, but cuts the corpus hard and biases toward reddit's taste.

---

## Stage checklist

- [x] 0 — Scaffolding, config, environment test
- [x] 0b — Real GPT-2 weights downloaded and verified
- [x] 1a — Datasets downloaded (1.23M raw jokes)
- [x] 1b — Cleaning pipeline written and tested (44 tests)
- [x] 1c — re-ran the cleaner with the fixed filters (568,476 jokes)
- [ ] **1d — fix noisy context labels ← START HERE**
- [ ] 2 — SFT: LoRA fine-tune (Colab notebook)
- [ ] 3 — Reward model: humour classifier + 4 other signals
- [ ] 4 — RL: GRPO with KL penalty (Colab notebook)
- [ ] 5 — Gradio A/B voting → DPO
- [ ] 6 — Evaluation: SFT vs RL vs RL+HF, reward curves

---

## Command reference

```bash
python -m src.test_setup              # env check (tiny model)
python -m src.test_setup --full       # env check (real gpt2)
python -m pytest tests/ -q            # 44 tests
python -m src.data_download           # fetch raw corpora (cached)
python -m src.data_clean              # clean + tag  (~15 min)
python -m src.data_clean --no-embed   # keyword tagging only (~2 min)
python -m src.data_clean --limit 20000 --no-embed   # fast iteration
HF_HUB_DISABLE_XET=1 python ...       # needed for large HF downloads here
```
