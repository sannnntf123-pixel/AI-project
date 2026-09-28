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

### ✅ Stage 1 — Data (code done, one re-run outstanding)

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

## ⚠️ OPEN TASK 1c — re-run the cleaner (START HERE)

**`data/processed/` currently holds output from the FIRST run, which has two
known defects.** The fixes are written, tested and committed — the corpus just
has not been regenerated. The re-run was interrupted partway.

```bash
python -m src.data_clean          # ~15 min, mostly embedding-tagging on MPS
```

What the fixes address, both found by auditing the output:

1. **207 jokes using the British spelling of a slur** survived, because the
   wordlist only had the American spelling. Digit-substituted variants
   (`n1gger`, `f4ggot`) also got through. Both now covered, with regression
   tests. **This matters for a graded university project** — check the output
   before submitting.
2. **4,405 reddit meta-posts** are not jokes at all (joke-request threads,
   karma complaints, announcements). A new `is_meta_post()` filter removes them.

Expect roughly **570,000** jokes after the re-run, slightly fewer than 574,275.

Verify afterwards:
```bash
python -c "
import json,re
rows=[json.loads(l) for l in open('data/processed/jokes_train.jsonl')]
print('rows:', len(rows))
print('paedo leaks:', sum(1 for r in rows if re.search(r'paedo', r['joke'], re.I)))
print('meta leaks :', sum(1 for r in rows if re.search(r'joke thread|upvote', r['joke'], re.I)))
"
```

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
- [ ] **1c — re-run the cleaner with the fixed filters ← START HERE**
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
