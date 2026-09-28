"""Stage 1a — fetch the raw datasets.

Two sources, both from Hugging Face rather than Kaggle (the original Short Jokes
needs a Kaggle API token; this mirror does not):

  1. Short Jokes  -- 231K clean one-liners. No scores, so it is pure SFT
     material: it teaches the model what a joke looks like.

  2. r/Jokes (one-million-reddit-jokes) -- 1M reddit posts with **upvote
     scores**. The scores are the whole point: they are the only human signal
     of what is actually funny, and they become the training labels for the
     humour classifier in stage 3. Reddit jokes are split across two fields --
     `title` is the setup, `selftext` is the punchline -- so they need joining.

Downloads are cached by huggingface_hub, so re-running this is cheap and safe.

Run:  python -m src.data_download
      python -m src.data_download --reddit-rows 200000   # smaller sample
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import (
    PATHS,
    REDDIT_JOKES_REPO,
    SHORT_JOKES_FILE,
    SHORT_JOKES_REPO,
)

SHORT_JOKES_OUT = "short_jokes.csv"
REDDIT_JOKES_OUT = "reddit_jokes.csv"


def download_short_jokes(force: bool = False) -> pd.DataFrame:
    """Fetch the Short Jokes CSV (~24MB) and normalise its columns."""
    from huggingface_hub import hf_hub_download

    out = PATHS.raw / SHORT_JOKES_OUT
    if out.exists() and not force:
        print(f"  cached: {out}")
        return pd.read_csv(out)

    print(f"  downloading {SHORT_JOKES_REPO}/{SHORT_JOKES_FILE} ...")
    local = hf_hub_download(
        repo_id=SHORT_JOKES_REPO,
        filename=SHORT_JOKES_FILE,
        repo_type="dataset",
    )
    df = pd.read_csv(local)
    # Columns arrive as ID,Joke. Rename to the schema the cleaner expects, and
    # add the columns this source lacks so both sources share one shape.
    df = df.rename(columns={"Joke": "text", "ID": "source_id"})[["source_id", "text"]]
    df["score"] = pd.NA          # no upvote data in this source
    df["source"] = "short_jokes"
    df.to_csv(out, index=False)
    print(f"  saved {len(df):,} rows -> {out}")
    return df


def download_reddit_jokes(max_rows: int | None = None, force: bool = False) -> pd.DataFrame:
    """Fetch r/Jokes posts with their upvote scores.

    `max_rows` truncates the 1M-row set. Keep it None for the real run; a
    smaller number is useful while iterating on the cleaning code.
    """
    from datasets import load_dataset

    out = PATHS.raw / REDDIT_JOKES_OUT
    if out.exists() and not force:
        print(f"  cached: {out}")
        return pd.read_csv(out)

    print(f"  downloading {REDDIT_JOKES_REPO} ...")
    ds = load_dataset(REDDIT_JOKES_REPO, split="train")
    df = ds.to_pandas()
    print(f"  got {len(df):,} raw rows, columns: {list(df.columns)}")

    if max_rows is not None:
        # Take the highest-scoring rows rather than the first N: a random slice
        # of reddit is mostly low-effort posts, and the scores are what we came
        # for.
        df = df.nlargest(max_rows, "score")
        print(f"  kept top {len(df):,} by score")

    # A reddit joke is setup + punchline across two fields. Join them, because
    # the title alone is usually just the setup and makes no sense on its own.
    title = df["title"].fillna("").astype(str).str.strip()
    body = df["selftext"].fillna("").astype(str).str.strip()
    joined = (title + " " + body).str.strip()

    result = pd.DataFrame(
        {
            "source_id": df["id"].astype(str).values,
            "text": joined.values,
            "score": df["score"].values,
            "source": "reddit_jokes",
            # Keep reddit's own NSFW flag: it is a free, human-applied signal
            # that the offensive-content filter in stage 1b can use.
            "nsfw": df["subreddit.nsfw"].fillna(False).values,
        }
    )
    result.to_csv(out, index=False)
    print(f"  saved {len(result):,} rows -> {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Download raw joke datasets")
    ap.add_argument("--reddit-rows", type=int, default=None,
                    help="keep only the top-N highest-scoring reddit rows")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--skip-reddit", action="store_true")
    ap.add_argument("--skip-short", action="store_true")
    args = ap.parse_args()

    PATHS.ensure()
    print(f"raw data dir: {PATHS.raw}\n")

    frames = []
    if not args.skip_short:
        print("[1/2] Short Jokes")
        frames.append(download_short_jokes(force=args.force))
        print()
    if not args.skip_reddit:
        print("[2/2] r/Jokes (with upvote scores)")
        frames.append(download_reddit_jokes(max_rows=args.reddit_rows, force=args.force))
        print()

    total = sum(len(f) for f in frames)
    print("-" * 60)
    print(f"total raw jokes: {total:,}")
    for f in frames:
        src = f["source"].iloc[0]
        has_score = f["score"].notna().sum()
        print(f"  {src:<14} {len(f):>9,} rows, {has_score:>9,} with scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
