"""Central configuration for the joke-RL project.

Every other module imports its settings from here, so there is exactly one
place to change a model name, a reward weight, or an output path. Values are
grouped into frozen dataclasses by pipeline stage (data -> SFT -> reward -> RL)
so that related knobs stay together and nothing can be mutated by accident at
run time.

Usage:
    from src.config import PATHS, SFT, REWARD, RL, get_device
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
        return self.root / "data"

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
        return self.root / "models"

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
# gpt2 (124M) is the safe default: it trains on a MacBook Air in minutes and on
# a Colab T4 in well under an hour. Swap base_model for "Qwen/Qwen2.5-0.5B" on
# Colab if you want noticeably better jokes; nothing else needs to change.

@dataclass(frozen=True)
class SFTConfig:
    base_model: str = "gpt2"
    max_length: int = 128

    # LoRA: instead of updating all 124M weights, we train small rank-r adapter
    # matrices injected into the attention projections. ~0.5% of the parameters
    # are trainable, which is what makes this feasible without a big GPU.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Attention projection names differ per architecture:
    #   gpt2  -> "c_attn" (fused qkv)
    #   qwen2 -> "q_proj", "k_proj", "v_proj", "o_proj"
    # If you switch base_model, this MUST change too or LoRA silently attaches
    # to nothing.
    lora_target_modules: tuple[str, ...] = ("c_attn",)
    # GPT-2's c_attn is a Conv1D, whose weight matrix is stored transposed
    # relative to nn.Linear. PEFT detects this and flips fan_in_fan_out itself
    # (it prints a UserWarning saying so) -- the warning is expected on gpt2 and
    # will not appear on Qwen, which uses real Linear layers.

    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 25
    save_steps: int = 500


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
