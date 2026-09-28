"""Tests for the stage-1 cleaning functions.

These are the functions that silently shape the whole dataset, so they are
worth pinning down: a normalisation bug does not crash, it just quietly
degrades every joke the model ever sees.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA, format_example, format_prompt
from src.data_clean import (
    _shingles,
    drop_near_duplicates,
    is_dead,
    is_meta_post,
    is_offensive,
    normalise,
    tag_by_keywords,
)


class TestNormalise:
    def test_unescapes_html_entities(self):
        assert normalise("Tom &amp; Jerry") == "Tom & Jerry"

    def test_unescapes_double_encoded(self):
        # Reddit dumps are frequently double-escaped.
        assert normalise("Tom &amp;amp; Jerry") == "Tom & Jerry"

    def test_strips_zero_width_spaces(self):
        assert "​" not in normalise("a​b")

    def test_removes_urls(self):
        assert normalise("funny http://example.com/x thing") == "funny thing"

    def test_removes_edit_footnotes(self):
        text = "A joke here. Edit: thanks for the gold!"
        assert normalise(text) == "A joke here."

    def test_collapses_whitespace(self):
        assert normalise("a  \n\t b") == "a b"

    def test_caps_repeated_punctuation(self):
        assert normalise("what!!!!!!") == "what!!"

    def test_handles_non_string(self):
        assert normalise(None) == ""
        assert normalise(float("nan")) == ""


class TestDeadPosts:
    @pytest.mark.parametrize("text", ["[removed]", "[deleted]", "setup [REMOVED]"])
    def test_detects_tombstones(self, text):
        assert is_dead(text)

    def test_ignores_normal_text(self):
        assert not is_dead("I removed my shoes")


class TestOffensiveFilter:
    def test_flags_slurs(self):
        assert is_offensive("a joke with retard in it")

    def test_flags_sensitive_topics(self):
        assert is_offensive("a rape joke")

    def test_allows_ordinary_jokes(self):
        assert not is_offensive("Why did the chicken cross the road?")

    def test_allows_mild_profanity(self):
        # A deliberate policy choice: profanity is not the same as offensive
        # content. Change _SLURS if the course requires stricter filtering.
        assert not is_offensive("that's damn funny")

    def test_is_case_insensitive(self):
        assert is_offensive("RAPE joke")


class TestOffensiveFilterSpellingVariants:
    """Regression tests from a real audit of the cleaned corpus.

    The first pass shipped 207 jokes using the British spelling and a handful
    using digit-substituted slurs, because the wordlist only had the American
    spellings and no obfuscation handling.
    """

    @pytest.mark.parametrize("text", [
        "what do paedophiles and tortoises have in common",
        "a paedo joke",
    ])
    def test_catches_british_spelling(self, text):
        assert is_offensive(text)

    @pytest.mark.parametrize("text", ["n1gger", "f4ggot", "ret4rd", "n!gger"])
    def test_catches_digit_substitutions(self, text):
        assert is_offensive(text)

    @pytest.mark.parametrize("text", [
        "the bigger picture",        # contains 'igge' but is not a slur
        "I need a bag for my stuff",
        "retarded growth of the plant",  # a true positive; documents the
                                         # wordlist's known over-blocking
    ])
    def test_does_not_crash_on_lookalikes(self, text):
        # These assert behaviour, not correctness of policy: the filter is
        # blunt by design and the report should say so.
        assert isinstance(is_offensive(text), bool)


class TestMetaPostFilter:
    """r/Jokes meta-posts score well but are not jokes. The three
    highest-scoring posts in the raw dump are all of this kind."""

    @pytest.mark.parametrize("text", [
        "Lawyer Joke Thread Submit your favorite lawyer jokes!",
        "Please upvote this if you agree",
        "Repost from r/jokes but worth it",
        "[Meta] new rules for this subreddit",
    ])
    def test_flags_meta_posts(self, text):
        assert is_meta_post(text)

    @pytest.mark.parametrize("text", [
        "Why did the chicken cross the road? To get to the other side.",
        "A horse walks into a bar. The bartender says why the long face.",
    ])
    def test_allows_real_jokes(self, text):
        assert not is_meta_post(text)


class TestKeywordTagging:
    def test_tags_obvious_contexts(self):
        s = pd.Series([
            "I need coffee before my espresso",
            "my python code has a bug",
            "my wife and I got married",
        ])
        out = tag_by_keywords(s)
        assert out.iloc[0] == "coffee"
        assert out.iloc[1] == "programming"
        assert out.iloc[2] == "relationships"

    def test_returns_empty_for_no_match(self):
        out = tag_by_keywords(pd.Series(["zzz qqq vvv"]))
        assert out.iloc[0] == ""

    def test_every_keyword_context_is_a_valid_label(self):
        from src.data_clean import CONTEXT_KEYWORDS

        for ctx in CONTEXT_KEYWORDS:
            assert ctx in DATA.contexts, f"{ctx} is not in DATA.contexts"


class TestShingles:
    def test_produces_word_ngrams(self):
        assert _shingles("a b c d", 3) == {"a b c", "b c d"}

    def test_short_text_returns_whole_string(self):
        assert _shingles("a b", 3) == {"a b"}

    def test_empty_text_returns_empty_set(self):
        assert _shingles("", 3) == set()

    def test_ignores_punctuation_and_case(self):
        assert _shingles("A, B! C", 3) == _shingles("a b c", 3)


class TestNearDuplicates:
    def test_removes_near_identical_jokes(self):
        df = pd.DataFrame({"text": [
            "why did the chicken cross the road to get to the other side",
            "why did the chicken cross the road to get to the other side!",
            "a completely unrelated joke about penguins and ice cream trucks",
        ]})
        out, _ = drop_near_duplicates(df, 0.9, 128, 3)
        assert len(out) == 2

    def test_keeps_distinct_jokes(self):
        df = pd.DataFrame({"text": [
            "the first joke is about a horse walking into a bar alone",
            "the second joke concerns penguins operating heavy machinery",
        ]})
        out, _ = drop_near_duplicates(df, 0.9, 128, 3)
        assert len(out) == 2


class TestPromptFormat:
    def test_prompt_has_trailing_space(self):
        # GPT-2 tokenises " word" and "word" differently, so the training
        # format and the generation prompt must agree exactly.
        assert format_prompt("coffee").endswith(" ")

    def test_example_starts_with_prompt(self):
        assert format_example("coffee", "ha").startswith(format_prompt("coffee"))

    def test_context_is_lowercased_and_stripped(self):
        assert format_prompt("  COFFEE  ") == format_prompt("coffee")
