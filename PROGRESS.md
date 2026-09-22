# PROGRESS — where the project stands

Update this file at the end of every working session. It is the first thing to
read when picking the project back up. Architecture and conventions live in
`CLAUDE.md`; this file only tracks **what is done and what is next**.

---

## Last session — 2026-09-22

### Stage 0 — Scaffolding ✅ DONE

Committed as `4f69ff33` and pushed to `origin/main`.

| File | What it is |
|---|---|
| `src/config.py` | Single source of truth. Frozen dataclasses per stage: `PATHS, DATA, SFT, REWARD, RL, DPO, EVAL, APP`. Helpers: `format_prompt()`, `format_example()`, `get_device()`, `get_dtype()`, `use_fp16()`. |
| `src/test_setup.py` | Environment smoke test. `python -m src.test_setup` |
| `.gitignore` | Excludes `.venv/`, `data/*`, `models/*`; keeps folders via `.gitkeep`. |
| `README.md` | Project writeup with pipeline diagram. |
| `CLAUDE.md` | Working context and constraints. |
| `requirements.txt` | Direct deps. Full freeze in `requirements-lock.txt`. |

### Verified working
- Python 3.11.9 arm64, all 11 dependencies import.
- **MPS available**, real matmul runs on it. No CUDA locally (expected).
- Full model path: load → tokenizer pad-token fix → **LoRA wrap on `c_attn`** →
  `generate()`. Exercised against `sshleifer/tiny-gpt2` and it passed.

### Two things resolved that will bite again if forgotten

**1. The repo originally had one commit containing only the 443MB `.venv`.**
It could not be pushed — GitHub rejects files over 100MB and it held a 368MB
`libtorch_cpu.dylib`. Fixed by clearing the index (`git read-tree --empty`),
re-adding under the new `.gitignore`, and garbage-collecting: **434MB → 124KB**.
Note that `git rm --cached` alone does *not* fix this — the blob stays in
history. **Never `git add` the `.venv` again.**

**2. The network here is very slow (~57 KB/s to the HF CDN).**
GPT-2's `model.safetensors` is 548MB ≈ 2.7 hours. That is why `--tiny` exists.

### ⚠️ Unfinished from this session
The real **GPT-2 download was still in progress** (~12MB of 548MB) when the
session ended, so `test_setup.py` has only been verified against the tiny
stand-in. The tiny model shares GPT-2's architecture and tokenizer, so the code
path is genuinely validated — but confirm with the real weights:

```bash
python -m src.test_setup          # full check, needs gpt2 downloaded
python -m src.test_setup --tiny   # fast check, no large download
```

If the download did not finish, resume it in the background and carry on with
data work, which needs no model:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('gpt2')"
```

---

## ▶ NEXT SESSION STARTS HERE — Stage 1: data

Goal: produce `data/processed/jokes.jsonl`, cleaned and context-tagged, ready
for SFT. **No GPU or model download needed** — good work to do while weights
pull in the background.

### Steps
1. **`src/data_download.py`** — fetch the two sources into `data/raw/`.
   - Short Jokes (Kaggle, ~231k rows). Kaggle needs an API token at
     `~/.kaggle/kaggle.json` — it is gitignored; set it up before starting.
     Check whether a Hugging Face mirror avoids the Kaggle auth entirely.
   - r/Jokes **with upvote scores** — the scores are not optional, they are the
     training labels for the humour classifier in stage 3.
2. **`src/data_clean.py`** — the real work:
   - drop HTML/markdown escapes, URLs, `[removed]`/`[deleted]`, encoding junk
   - length filter: `DATA.min_words`=4 … `DATA.max_words`=60, `max_chars`=400
   - exact dedupe, then **near-dedupe at TF-IDF cosine > `DATA.dedup_threshold`
     (0.90)**. On ~230k rows a naive all-pairs comparison is ~26 billion
     comparisons — do not write that loop. Use sparse matrix multiplication in
     blocks, or MinHash/LSH via `datasketch`.
   - offensive-content filter (keep the dropped rows for the report — the count
     is a result worth stating)
3. **Context tagging** — assign one of `DATA.contexts` (15 labels) per joke.
   Start with keyword rules, then consider zero-shot embedding similarity via
   sentence-transformers for the leftovers. Expect a large `general` bucket;
   report the class distribution.
4. **Write out** using `config.format_example()` — never rebuild the
   `<context> … <joke> …` string by hand anywhere.
5. **Sanity check**: print 20 random tagged examples and read them. Confirm the
   split with `DATA.test_size` (0.05) and `DATA.seed` (42).

### Decisions still open
- Which r/Jokes dump to use, and how to map upvotes to a binary funny/not label
  for the classifier (`DATA.reddit_min_score` = 100 is a placeholder guess).
- Whether the offensive filter is a wordlist or a model. A model is slower but
  far more defensible in the writeup.

---

## Stage checklist

- [x] 0 — Scaffolding, config, environment test
- [ ] 1 — Data: download, clean, dedupe, filter, context-tag  ← **next**
- [ ] 2 — SFT: LoRA fine-tune (Colab notebook)
- [ ] 3 — Reward model: humour classifier + 4 other signals
- [ ] 4 — RL: GRPO with KL penalty (Colab notebook)
- [ ] 5 — Gradio A/B voting → DPO
- [ ] 6 — Evaluation: SFT vs RL vs RL+HF, reward curves
