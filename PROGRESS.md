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
| `src/model_io.py` | `load_policy()` / `save_policy()` / `resolve_checkpoint()`. Handles LoRA adapters and full models, searches `dpo -> rl -> sft`, falls back to the untrained base. |
| `.env.example` | Every environment variable, documented. |
| `.gitignore` | Excludes `.venv/`, `data/*`, `models/*`; keeps folders via `.gitkeep`. |
| `README.md` | Project writeup with pipeline diagram. |
| `CLAUDE.md` | Working context and constraints. |
| `requirements.txt` | Direct deps. Full freeze in `requirements-lock.txt`. |

### Profile system (added after the first commit)

`PROFILE` is `local` or `colab`, auto-detected, and decides the model:

- **local** → `sshleifer/tiny-gpt2`, ~100K params, gibberish output by design,
  1 epoch, 200 training samples. For checking that code runs.
- **colab** → `gpt2`, full training.

LoRA target modules are derived from the model id via `lora_targets_for()`,
**not hardcoded** — so switching to Qwen on Colab picks up `q_proj`/`k_proj`/
`v_proj`/`o_proj` automatically. Hardcoding them is a silent failure: PEFT
attaches to nothing and the loss barely moves.

`PATHS.models` and `PATHS.data` honour `JOKE_RL_MODELS_DIR` / `JOKE_RL_DATA_DIR`,
so Colab training writes to Drive (surviving a recycled runtime) and the local
Gradio app can serve that same checkpoint with no code change.

**Because of this, task 0b matters even less** — the tiny model is now the local
default, so nothing here needs the 548MB GPT-2 download.

### Verified working
- Python 3.11.9 arm64, all 11 dependencies import.
- **MPS available**, real matmul runs on it. No CUDA locally (expected).
- Full model path: load → tokenizer pad-token fix → **LoRA wrap on `c_attn`** →
  `generate()`. Exercised against `sshleifer/tiny-gpt2` and it passed.
- **Save/load round trip**: built a LoRA model, `save_policy()` → `load_policy()`
  → generated from the reloaded checkpoint. The adapter records its own base
  model, so loading needs no extra arguments.
- **Drive scenario**: a checkpoint in an outside directory was found and loaded
  purely via `JOKE_RL_MODELS_DIR`, and pinned exactly via `JOKE_RL_APP_MODEL`.
  Bad paths raise a clear error rather than silently loading the base model.

### Two things resolved that will bite again if forgotten

**1. The repo originally had one commit containing only the 443MB `.venv`.**
It could not be pushed — GitHub rejects files over 100MB and it held a 368MB
`libtorch_cpu.dylib`. Fixed by clearing the index (`git read-tree --empty`),
re-adding under the new `.gitignore`, and garbage-collecting: **434MB → 124KB**.
Note that `git rm --cached` alone does *not* fix this — the blob stays in
history. **Never `git add` the `.venv` again.**

**2. The network here is very slow (~57 KB/s to the HF CDN).**
GPT-2's `model.safetensors` is 548MB ≈ 2.7 hours. That is why `--tiny` exists.

### ⚠️ OPEN TASK — download the real GPT-2 weights (deferred, not blocking)

**Status: deliberately postponed. The network here is too slow right now.**

`test_setup.py` has only been verified against `sshleifer/tiny-gpt2`. That model
has GPT-2's real architecture and tokenizer, so the code path *is* genuinely
validated — LoRA attaches to `c_attn`, generation runs on MPS. What is not yet
verified is the real 548MB checkpoint.

**What was measured (2026-09-22):**

| Host | Sustained speed |
|---|---|
| HF CDN (`model.safetensors`) | **2.9 KB/s** — effectively stalled |
| GitHub | ~217 KB/s |
| PyPI / Fastly | ~58 KB/s |

`huggingface_hub` retried three times and wrote **0 bytes** each time; a
resumable `curl` managed 172KB in 60s. At that rate 548MB needs ~50 hours, so
the download was stopped and the partial blob deleted (a truncated
`.safetensors` left at the real cache path makes `transformers` fail with a
confusing parse error rather than re-downloading).

**To do it later, on a better connection:**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('gpt2')"
python -m src.test_setup        # no --tiny: confirms the real weights
```

If the CDN throttles again, a resumable curl survives drops better than
`huggingface_hub` does:

```bash
BLOB=~/.cache/huggingface/hub/models--gpt2/blobs/248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
curl -L -C - --retry 10 -o "$BLOB" https://huggingface.co/gpt2/resolve/main/model.safetensors
```
(`-C -` resumes from wherever it stopped; re-run until the file reaches
548,105,171 bytes, then symlink it as `model.safetensors` in the
`snapshots/<hash>/` directory.)

**Colab has fast internet.** The simplest path may be to skip the local
download entirely and let the Colab notebook in stage 2 pull GPT-2 itself —
which is where the real training happens anyway. Nothing before stage 2 needs
the weights.

**This does not block anything.** Stage 1 (data) needs no model at all.

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
- [ ] **0b — download real GPT-2 weights** (deferred: network too slow; not blocking)
- [ ] 1 — Data: download, clean, dedupe, filter, context-tag  ← **next**
- [ ] 2 — SFT: LoRA fine-tune (Colab notebook)
- [ ] 3 — Reward model: humour classifier + 4 other signals
- [ ] 4 — RL: GRPO with KL penalty (Colab notebook)
- [ ] 5 — Gradio A/B voting → DPO
- [ ] 6 — Evaluation: SFT vs RL vs RL+HF, reward curves
