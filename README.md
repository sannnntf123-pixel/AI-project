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
Short Jokes and r/Jokes are merged, stripped of duplicates and near-duplicates
(TF-IDF cosine > 0.90), filtered for offensive content, and each joke is tagged
with one of 15 context labels. r/Jokes upvote scores double as the supervision
signal for the humour classifier in stage 3.

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
