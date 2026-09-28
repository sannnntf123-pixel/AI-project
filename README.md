# Context-Conditioned Joke Generation with Reinforcement Learning

A language model that writes jokes about a topic you choose — `coffee`,
`university exams`, `programming` — trained with reinforcement learning against
a composite reward, then refined with human preference feedback.

University AI course project. Every model here is trained from open weights in
this repository; no external LLM API generates any joke.

---

## The idea

Supervised fine-tuning teaches a model what a joke *looks like*. It cannot teach
it what a joke *lands*, because the training objective only rewards predicting
the next token of jokes that humans already wrote.

Reinforcement learning can optimise the thing we actually care about. The
difficulty is that "funny" has no loss function, so the project builds one: a
reward model that scores a generated joke on five separate axes, and an RL loop
that pushes the policy up that score while a KL penalty stops it from wandering
off into text that games the reward without being a joke.

## Pipeline

```
   Short Jokes (Kaggle)          r/Jokes (with upvote scores)
            │                              │
            └──────────────┬───────────────┘
                           ▼
              1. clean · dedupe · filter · tag with context
                           │
                           ▼
           <context> coffee <joke> ...            ← training format
                           │
                           ▼
              2. SFT  —  LoRA fine-tune of GPT-2
                           │
                           ├──────────────► frozen reference (KL anchor)
                           ▼
              4. RL  —  GRPO, TRL                 ◄── 3. reward model
                           │                            humor    0.50
                           ▼                            relevance 0.25
              5. Gradio A/B voting → DPO                novelty   0.15
                           │                            toxicity  0.10
                           ▼                            length    0.00
              6. evaluation: SFT vs RL vs RL+HF
```

### 1. Data

Both corpora come from Hugging Face, not Kaggle — the original Short Jokes needs
a Kaggle API token, and `Fraser/short-jokes` is a loading script, which
`datasets` 5.x no longer executes.

| Source | Rows | Scores |
|---|---|---|
| `ysharma/short_jokes` | 231,657 | no |
| `SocialGrep/one-million-reddit-jokes` | 1,000,000 | **yes** |

The upvote scores are the point of the reddit set: they are the only human
signal of what is actually funny, and they become the labels for the humour
classifier in stage 3. Reddit splits a joke across two fields — `title` is the
setup, `selftext` the punchline — so they are joined before anything else.

Cleaning runs cheapest-filter-first:

```
normalise → drop tombstones → length → exact dedup → near-dedup
  → offensive → meta-posts → context tagging → split
```

**Near-duplicate removal uses MinHash + LSH**, not all-pairs TF-IDF cosine. At
1.2M rows the naive comparison is ~7×10¹¹ pairs. MinHash compresses each joke's
word-3-shingle set into a signature whose collision probability equals the
Jaccard similarity; LSH buckets signatures so only plausible matches are ever
compared. Rows are sorted by score first, so the highest-scoring copy of a
repeated joke is the one kept.

**Context tagging is two-pass**: keyword rules first (fast, deterministic, easy
to justify — about 29% coverage), then zero-shot sentence-embedding similarity
against `"a joke about X"` for the rest, falling back to `general` below a
similarity floor rather than forcing a bad label.

Two findings from auditing the output, both worth reporting:

- **High reddit score does not mean funny.** The three highest-scoring posts in
  the raw dump are an obituary, a net-neutrality protest, and a
  broken-keyboard bit. r/Jokes upvotes community drama as readily as jokes, so
  a meta-post filter runs before the score is ever used as a humour label.
- **A wordlist filter misses what it was not told about.** The first pass let
  through 207 jokes using the British spelling of a slur the list only had in
  American form, plus digit-substituted variants written specifically to evade
  filters. Both are now covered, with regression tests.

### 2. Supervised fine-tuning
LoRA adapters on a small pretrained model — `gpt2` (124M) by default, so it
trains in minutes on a laptop. Roughly 0.5% of parameters are trainable, which
is what makes the whole project feasible without a dedicated GPU. This stage
teaches the format: given `<context> coffee <joke>`, continue with a joke about
coffee.

### 3. Reward model
A weighted sum of five signals rather than one learned score:

| Signal | Weight | What it measures |
|---|---|---|
| **humor** | 0.50 | DistilBERT classifier trained here on upvote-labelled jokes |
| **relevance** | 0.25 | sentence-transformers cosine similarity between context and joke |
| **novelty** | 0.15 | penalty for near-copies of training jokes |
| **toxicity** | 0.10 | penalty, via a toxicity classifier |
| **length** | 0.00 | soft penalty outside 8–40 words (off by default) |

The composition is the point. A single learned humour head is easy for the
policy to game — it will find one strange string that scores 0.99 and emit it
forever. The other four terms exist to close off those exploits, and the gap
between them is itself a result worth reporting.

### 4. Reinforcement learning
**GRPO** via Hugging Face TRL. GRPO samples a group of jokes per prompt and
pushes the policy toward the above-average ones, which means it needs no
separate value network — half the memory of PPO and far fewer things to tune on
a free Colab T4. A **KL penalty against the frozen SFT model** is the main guard
against reward hacking.

### 5. Human feedback
A Gradio app shows two jokes for a context and asks which is funnier. Votes
accumulate as preference pairs and feed **DPO**, or retrain the reward model.

### 6. Evaluation
`SFT` vs `RL` vs `RL + human feedback`, on humour score, context relevance,
novelty, diversity (distinct-1/2/3), and blind human ratings — plus reward
curves over training.

---

## Setup

```bash
git clone https://github.com/sannnntf123-pixel/AI-project.git
cd AI-project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the environment — checks the interpreter, every dependency, the compute
backend, and does a real GPT-2 load-and-generate:

```bash
python -m src.test_setup
```

## Configuration

Everything is driven by [`src/config.py`](src/config.py) plus a few environment
variables. See [`.env.example`](.env.example).

### Two profiles

The same code runs a throwaway model locally and the real one on Colab:

| Profile | Model | Purpose |
|---|---|---|
| `local` | `sshleifer/tiny-gpt2` (~100K params) | Prove the code runs. Output is gibberish — the model is randomly initialised. Downloads in seconds. |
| `colab` | `gpt2` (124M) | Actual training. |

The profile is auto-detected — `colab` inside Google Colab, `local` otherwise —
so normally you set nothing. Override when you need to:

```bash
JOKE_RL_PROFILE=colab python -m src.test_setup     # force the real model
JOKE_RL_BASE_MODEL=distilgpt2 python -m src.train  # one-off model override
```

Both models share GPT-2's architecture and tokenizer, so nothing else changes.
LoRA target module names are resolved *from the model id* by
`lora_targets_for()` rather than hardcoded — switching to `Qwen/Qwen2.5-0.5B`
picks up `q_proj`/`k_proj`/`v_proj`/`o_proj` automatically. This matters because
wrong target names are a silent failure: PEFT attaches to nothing, training
runs, and the loss barely moves.

Locally `SFT.effective_epochs` is 1 and `SFT.max_train_samples` is 200, so a
training smoke test finishes in seconds instead of minutes.

### Loading models from Google Drive

Colab runtimes are recycled without warning, so training writes to Drive and
the local app reads from it. One variable redirects both:

```python
# top of a Colab notebook
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ["JOKE_RL_MODELS_DIR"] = "/content/drive/MyDrive/joke-rl/models"
```

```bash
# locally, against a synced Drive folder
JOKE_RL_MODELS_DIR="$HOME/Google Drive/joke-rl/models" python -m app.demo
```

`model_io.load_policy()` then finds the best checkpoint on its own, preferring
`dpo` → `rl` → `sft`, and falling back to the untrained base model so the app is
runnable before any training exists:

```python
from src.model_io import load_policy
model, tokenizer, info = load_policy()        # best available
model, tokenizer, info = load_policy("sft")   # a specific stage
print(info)   # "rl (LoRA adapter) on gpt2 — /content/drive/.../models/rl"
```

It handles LoRA adapters and full models, reads the base model out of
`adapter_config.json` so you never have to remember it, and can merge adapters
for faster inference.

## Layout

```
data/        raw and processed datasets   (gitignored)
models/      trained checkpoints          (gitignored)
notebooks/   Colab notebooks for the heavy training stages
src/         library code — config, data, training, reward, eval
app/         Gradio demo and human-feedback collection
tests/       tests
outputs/     plots and evaluation tables  (gitignored)
```

All configuration — model names, reward weights, hyperparameters, paths — lives
in [`src/config.py`](src/config.py). Nothing else hardcodes a setting.

## Hardware

Developed on a MacBook Air (Apple Silicon, MPS); heavy training runs on Google
Colab (T4). The same scripts run on both — `config.get_device()` resolves
CUDA → MPS → CPU, and precision follows the backend.

## Status

- [x] Scaffolding, configuration, environment test
- [ ] Data collection and cleaning
- [ ] Supervised fine-tuning
- [ ] Reward model
- [ ] GRPO training
- [ ] Gradio human feedback + DPO
- [ ] Evaluation

## Licence and data

Code is for coursework. Short Jokes and r/Jokes are used under their respective
dataset licences; neither is redistributed in this repository.
