"""The checks that do not trust the model."""

from tailor import check_quote, unsupported_terms
from tailor.verify import normalize

SOURCES = {
    "a.md:1-1": "Built a Python pipeline with pandas that cleans 5,000+ records from MongoDB.",
    "b.md:2-2": "Assessed one of Mongolia’s largest banks against the SOC-CMM framework.",
}


def test_quotes_must_appear_word_for_word():
    assert check_quote("cleans 5,000+ records", ["a.md:1-1"], SOURCES)
    assert not check_quote("cleans 6,000 records", ["a.md:1-1"], SOURCES)
    assert (
        "not in the cited evidence"
        in check_quote("cleans 6,000 records", ["a.md:1-1"], SOURCES).reason
    )


def test_normalization_forgives_whitespace_and_curly_quotes():
    assert check_quote("one of Mongolia's  largest\nbanks", ["b.md:2-2"], SOURCES)
    assert normalize("A’s  “B” — c") == 'a\'s "b" - c'


def test_ellipsis_joins_parts_in_order():
    assert check_quote("Built a Python pipeline ... from MongoDB", ["a.md:1-1"], SOURCES)
    assert not check_quote("from MongoDB ... Built a Python pipeline", ["a.md:1-1"], SOURCES)


def test_citations_are_checked_before_quotes():
    assert check_quote("", ["a.md:1-1"], SOURCES).reason == "no quote given"
    assert check_quote("Built", [], SOURCES).reason == "no evidence cited"
    assert "does not exist: z.md:9-9" in check_quote("Built", ["z.md:9-9"], SOURCES).reason
    # any one of several citations may hold the quote
    assert check_quote("SOC-CMM framework", ["a.md:1-1", "b.md:2-2"], SOURCES)


def test_unsupported_numbers_and_names_are_flagged():
    evidence = list(SOURCES.values())
    draft = (
        "I cleaned 5,000+ records at TalentSnipe and led 12 engineers at Globex. "
        "Python was the tool; the bank was in Mongolia."
    )
    flags = unsupported_terms(draft, evidence, allow=["TalentSnipe"])
    assert flags == ["12", "Globex"]


def test_sentence_starts_and_common_words_do_not_count():
    flags = unsupported_terms("Before that, I wrote Python. The bank passed.", ["Python bank"])
    assert flags == []
    assert unsupported_terms("I moved to New York in 2024.", ["Moved in 2024"]) == ["New York"]
