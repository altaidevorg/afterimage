"""Meta-prompt diversity helpers."""

from afterimage.simula.meta_prompt import pairwise_trigram_overlap


def test_trigram_overlap_identical():
    t = "the quick brown fox jumps"
    assert pairwise_trigram_overlap(t, t) == 1.0


def test_trigram_overlap_disjoint():
    a = "aaa bbb ccc"
    b = "xyz qwe rty"
    assert pairwise_trigram_overlap(a, b) < 0.2
