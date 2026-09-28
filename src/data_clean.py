"""Stage 1b — clean, deduplicate, filter and context-tag the raw jokes.

Input : data/raw/{short_jokes,reddit_jokes}.csv   (from data_download.py)
Output: data/processed/jokes_{train,test}.jsonl   + a stats report

The pipeline, in order, because the order matters for speed:

  1. normalise text        (html entities, unicode, whitespace, urls)
  2. drop reddit artifacts ([removed], [deleted], empty)
  3. length filter         -- cheap, and removes ~half the rows before the
                              expensive steps run
  4. exact dedup
  5. near-dedup            -- MinHash + LSH
  6. offensive filter
  7. context tagging       -- keyword rules, then embeddings for the rest
  8. split and write

Run:  python -m src.data_clean
      python -m src.data_clean --no-embed     # keyword tagging only, faster
      python -m src.data_clean --limit 50000  # quick iteration
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter

import pandas as pd

from src.config import DATA, PATHS, REWARD, format_example

# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+|www\.\S+")
_REDDIT_DEAD = re.compile(r"\[(removed|deleted)\]", re.I)
_EDIT_NOTE = re.compile(r"\n*\**\s*edit\s*\d*\s*:.*$", re.I | re.S)
_WHITESPACE = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([!?.,])\1{2,}")


def normalise(text: str) -> str:
    """Turn a raw scraped string into clean joke text.

    Reddit text arrives HTML-escaped (`&amp;`), sometimes double-escaped, with
    zero-width spaces (`&#x200B;`) and "Edit:" footnotes that are commentary
    rather than part of the joke.
    """
    if not isinstance(text, str):
        return ""

    # Unescape twice: reddit dumps are frequently double-encoded.
    text = html.unescape(html.unescape(text))
    # NFKC folds exotic unicode lookalikes into plain ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", " ").replace(" ", " ")
    text = _URL.sub("", text)
    text = _EDIT_NOTE.sub("", text)
    text = _REPEAT_PUNCT.sub(r"\1\1", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def is_dead(text: str) -> bool:
    """Reddit tombstones: the post body was removed by a mod or the author."""
    return bool(_REDDIT_DEAD.search(text))


# --------------------------------------------------------------------------
# Offensive-content filter
# --------------------------------------------------------------------------
# A wordlist, not a model. The tradeoff, stated plainly because it belongs in
# the report: a wordlist is fast (one regex over 1M rows), fully transparent,
# and reproducible -- you can point at exactly why a joke was dropped. It is
# also blunt: it cannot read context, so it over-blocks reclaimed usage and
# under-blocks cruelty phrased politely.
#
# A transformer classifier (REWARD.toxicity_model) is the better filter, but
# running it over ~1M rows is hours of compute for a preprocessing step. The
# compromise used here: wordlist at data-cleaning time for the obvious cases,
# and the real toxicity model at RL time, where it scores only the few thousand
# jokes the policy actually generates and where it genuinely matters.

_SLURS = [
    # Racial / ethnic / religious slurs
    "nigger", "nigga", "chink", "gook", "spic", "wetback", "kike", "raghead",
    "towelhead", "paki", "coon", "beaner", "jap ", "wop ", "dago",
    # Homophobic / transphobic slurs
    "faggot", "fag ", "dyke", "tranny", "shemale",
    # Ableist slurs
    "retard", "retarded", "spastic", "mongoloid",
    # Misogynistic slurs
    "cunt", "whore", "slut",
]

_SENSITIVE_TOPICS = [
    # Sexual content involving minors -- zero tolerance, no context saves these
    "pedophile", "pedo ", "child porn", "underage sex", "molest",
    # Sexual violence
    "rape", "raping", "rapist",
    # Self-harm and extreme violence played for laughs
    "suicide", "kill yourself", "kys ",
    # Atrocity
    "holocaust", "genocide", "lynching",
]

_OFFENSIVE = re.compile(
    r"\b(" + "|".join(re.escape(w.strip()) for w in _SLURS + _SENSITIVE_TOPICS) + r")",
    re.I,
)


def is_offensive(text: str) -> bool:
    return bool(_OFFENSIVE.search(text))


# --------------------------------------------------------------------------
# Context tagging
# --------------------------------------------------------------------------
# Keyword rules first: they are fast, deterministic and easy to justify. Each
# context gets a list of indicative terms; the context with the most hits wins.
# Anything with no hits falls through to embedding similarity (or "general").

CONTEXT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "coffee": ("coffee", "espresso", "latte", "caffeine", "barista", "starbucks", "decaf", "cappuccino"),
    "university exams": ("exam", "midterm", "final exam", "study", "studying", "professor", "lecture",
                         "semester", "thesis", "dissertation", "graduate", "campus", "dorm", "gpa"),
    "programming": ("program", "code", "coding", "developer", "python", "javascript", "bug", "debug",
                    "compiler", "software", "git ", "database", "algorithm", "stack overflow", "api"),
    "work": ("boss", "office", "job", "meeting", "salary", "colleague", "manager", "coworker",
             "interview", "resume", "promotion", "overtime", "email"),
    "food": ("pizza", "burger", "sandwich", "cheese", "bread", "cook", "restaurant", "chef",
             "dinner", "breakfast", "lunch", "bacon", "egg", "chicken", "taco"),
    "animals": ("dog", "cat", "horse", "cow", "chicken", "bird", "fish", "bear", "lion",
                "elephant", "penguin", "duck", "rabbit", "snake", "puppy", "kitten"),
    "relationships": ("wife", "husband", "girlfriend", "boyfriend", "marriage", "married", "divorce",
                      "dating", "tinder", "date night", "ex-wife", "in-laws", "romance"),
    "technology": ("computer", "phone", "internet", "wifi", "robot", "ai ", "tech", "app ",
                   "laptop", "battery", "smartphone", "google", "facebook"),
    "sports": ("football", "soccer", "basketball", "baseball", "tennis", "golf", "hockey",
               "coach", "referee", "olympic", "marathon", "gym"),
    "money": ("money", "bank", "rich", "poor", "broke", "dollar", "loan", "debt", "tax",
              "wallet", "salary", "invest", "bitcoin", "credit card"),
    "school": ("school", "teacher", "classroom", "homework", "student", "principal", "grade",
               "kindergarten", "pupil", "detention"),
    "travel": ("flight", "airplane", "airport", "hotel", "vacation", "tourist", "luggage",
               "passport", "beach", "cruise", "train station"),
    "health": ("doctor", "hospital", "nurse", "medicine", "patient", "surgery", "dentist",
               "pill", "diet", "gym", "sick", "therapist"),
    "music": ("music", "guitar", "piano", "band", "singer", "song", "drummer", "concert",
              "violin", "album", "jazz", "rock band"),
}


def _build_keyword_regexes() -> dict[str, re.Pattern]:
    return {
        ctx: re.compile(r"\b(" + "|".join(re.escape(k.strip()) for k in kws) + r")", re.I)
        for ctx, kws in CONTEXT_KEYWORDS.items()
    }


def tag_by_keywords(texts: pd.Series) -> pd.Series:
    """Assign a context by counting keyword hits; empty string if none match.

    Vectorised with pandas' str.count rather than a Python loop -- on 600K rows
    the difference is minutes.
    """
    regexes = _build_keyword_regexes()
    scores = pd.DataFrame(
        {ctx: texts.str.count(rx) for ctx, rx in regexes.items()},
        index=texts.index,
    )
    best = scores.idxmax(axis=1)
    # idxmax returns a label even when every count is 0, so blank those out.
    return best.where(scores.max(axis=1) > 0, "")


def tag_by_embedding(texts: pd.Series, batch_size: int) -> pd.Series:
    """Assign a context by sentence-embedding similarity to the label names.

    This is zero-shot: we embed the 15 context labels and every untagged joke,
    then take the nearest label. Anything below DATA.context_min_similarity is
    left as "general" -- a weak match is worse than an honest default.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    from src.config import get_device

    model = SentenceTransformer(REWARD.similarity_model, device=get_device())
    labels = [c for c in DATA.contexts if c != "general"]
    # "a joke about X" embeds closer to joke text than the bare label does.
    label_emb = model.encode([f"a joke about {c}" for c in labels],
                             convert_to_tensor=True, normalize_embeddings=True)

    joke_emb = model.encode(texts.tolist(), batch_size=batch_size,
                            convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=True)
    sims = joke_emb @ label_emb.T
    best_sim, best_idx = torch.max(sims, dim=1)

    out = [
        labels[i] if s >= DATA.context_min_similarity else "general"
        for i, s in zip(best_idx.tolist(), best_sim.tolist())
    ]
    return pd.Series(out, index=texts.index)


# --------------------------------------------------------------------------
# Near-duplicate removal
# --------------------------------------------------------------------------

def _shingles(text: str, k: int) -> set[str]:
    """Word k-grams. Using words rather than characters makes the signature
    robust to punctuation and capitalisation differences."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def drop_near_duplicates(df: pd.DataFrame, threshold: float,
                         num_perm: int, k: int) -> tuple[pd.DataFrame, int]:
    """Remove near-duplicate jokes with MinHash + LSH.

    How it works, briefly: each joke becomes a set of word k-grams; MinHash
    compresses that set into a short signature whose collision probability
    equals the Jaccard similarity; LSH buckets signatures so that only likely
    matches are ever compared. That avoids the ~7e11 all-pairs comparisons a
    naive scan of 1.2M rows would need.

    We keep the *first* occurrence, and the frame is sorted so that the
    highest-scoring copy of a repeated joke is the one kept.
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[int] = []
    dropped = 0

    for pos, (idx, text) in enumerate(df["text"].items()):
        sh = _shingles(text, k)
        if not sh:
            dropped += 1
            continue
        m = MinHash(num_perm=num_perm)
        for s in sh:
            m.update(s.encode("utf8"))
        if lsh.query(m):
            dropped += 1
            continue
        lsh.insert(str(idx), m)
        keep.append(idx)
        if pos and pos % 100_000 == 0:
            print(f"      near-dedup {pos:,}/{len(df):,} — kept {len(keep):,}")

    return df.loc[keep], dropped


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def load_raw(limit: int | None = None) -> pd.DataFrame:
    frames = []
    for name in ("short_jokes.csv", "reddit_jokes.csv"):
        path = PATHS.raw / name
        if not path.exists():
            print(f"  ! missing {path} — run `python -m src.data_download` first")
            continue
        df = pd.read_csv(path)
        if limit:
            df = df.head(limit)
        frames.append(df[["source_id", "text", "score", "source"]])
        print(f"  loaded {len(df):>9,} from {name}")
    if not frames:
        raise SystemExit("No raw data found.")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, use_embeddings: bool) -> tuple[pd.DataFrame, dict]:
    stats: dict[str, int] = {"raw": len(df)}

    print("\n[1] normalising text")
    df["text"] = df["text"].map(normalise)

    print("[2] dropping reddit tombstones and empties")
    before = len(df)
    df = df[~df["text"].map(is_dead)]
    df = df[df["text"].str.strip() != ""]
    stats["dropped_dead_or_empty"] = before - len(df)
    print(f"    removed {stats['dropped_dead_or_empty']:,}")

    print("[3] length filter")
    before = len(df)
    wc = df["text"].str.split().str.len()
    df = df[(wc >= DATA.min_words) & (wc <= DATA.max_words)
            & (df["text"].str.len() <= DATA.max_chars)]
    stats["dropped_length"] = before - len(df)
    print(f"    removed {stats['dropped_length']:,} outside "
          f"{DATA.min_words}-{DATA.max_words} words / {DATA.max_chars} chars")

    print("[4] exact dedup")
    before = len(df)
    # Sort so the highest-scoring copy survives; NaN scores (short_jokes) last.
    df = df.sort_values("score", ascending=False, na_position="last")
    df = df.drop_duplicates(subset="text", keep="first")
    stats["dropped_exact_dup"] = before - len(df)
    print(f"    removed {stats['dropped_exact_dup']:,}")

    print(f"[5] near-dedup (MinHash LSH, Jaccard >= {DATA.dedup_threshold})")
    before = len(df)
    df, _ = drop_near_duplicates(df, DATA.dedup_threshold,
                                 DATA.minhash_perm, DATA.shingle_size)
    stats["dropped_near_dup"] = before - len(df)
    print(f"    removed {stats['dropped_near_dup']:,}")

    print("[6] offensive-content filter")
    before = len(df)
    offensive_mask = df["text"].map(is_offensive)
    stats["dropped_offensive"] = int(offensive_mask.sum())
    df = df[~offensive_mask]
    print(f"    removed {stats['dropped_offensive']:,} "
          f"({100*stats['dropped_offensive']/max(before,1):.2f}% of remaining)")

    print("[7] context tagging")
    df = df.reset_index(drop=True)
    df["context"] = tag_by_keywords(df["text"])
    tagged = (df["context"] != "").sum()
    print(f"    keyword rules tagged {tagged:,} / {len(df):,} "
          f"({100*tagged/max(len(df),1):.1f}%)")

    untagged = df["context"] == ""
    if untagged.any():
        if use_embeddings:
            print(f"    embedding-tagging the remaining {untagged.sum():,} ...")
            df.loc[untagged, "context"] = tag_by_embedding(
                df.loc[untagged, "text"], DATA.embed_batch_size)
        else:
            df.loc[untagged, "context"] = "general"
            print(f"    assigned {untagged.sum():,} to 'general' (--no-embed)")

    stats["final"] = len(df)
    return df, stats


def write_output(df: pd.DataFrame, stats: dict) -> None:
    from sklearn.model_selection import train_test_split

    PATHS.processed.mkdir(parents=True, exist_ok=True)

    train, test = train_test_split(
        df, test_size=DATA.test_size, random_state=DATA.seed,
        stratify=df["context"] if df["context"].nunique() > 1 else None,
    )

    for name, part in (("train", train), ("test", test)):
        path = PATHS.processed / f"jokes_{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in part.itertuples(index=False):
                fh.write(json.dumps({
                    "context": row.context,
                    "joke": row.text,
                    # The exact string the model trains on. Built by
                    # config.format_example so generation cannot disagree.
                    "text": format_example(row.context, row.text),
                    "score": None if pd.isna(row.score) else int(row.score),
                    "source": row.source,
                }, ensure_ascii=False) + "\n")
        print(f"  wrote {len(part):>8,} -> {path}")

    report = PATHS.processed / "clean_report.json"
    report.write_text(json.dumps({
        "stats": stats,
        "context_distribution": df["context"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "with_scores": int(df["score"].notna().sum()),
    }, indent=2))
    print(f"  wrote report -> {report}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean and tag the joke corpus")
    ap.add_argument("--limit", type=int, default=None,
                    help="only read N rows per source (for quick iteration)")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip embedding-based tagging; untagged -> 'general'")
    args = ap.parse_args()

    PATHS.ensure()
    print("loading raw data")
    df = load_raw(args.limit)

    df, stats = clean(df, use_embeddings=not args.no_embed)

    print("\n[8] writing output")
    write_output(df, stats)

    print("\n" + "=" * 62)
    print(f"{'raw':<26} {stats['raw']:>10,}")
    for k in ("dropped_dead_or_empty", "dropped_length", "dropped_exact_dup",
              "dropped_near_dup", "dropped_offensive"):
        print(f"{'  - ' + k[8:]:<26} {stats[k]:>10,}")
    print(f"{'final':<26} {stats['final']:>10,}  "
          f"({100*stats['final']/stats['raw']:.1f}% kept)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
