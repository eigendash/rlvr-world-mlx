"""The reward must be a metric on the decoded state, not on the token ids."""

from __future__ import annotations

import pytest

from rlvr_world.reward import (
    REWARD_SCHEMES,
    canonical_tokens,
    exact_match_reward,
    parse_state,
    reward_from_text,
    reward_from_tokens,
    score_rollouts,
    structural_reward,
    token_f1_reward,
)
from rlvr_world.tokenizer import Tokenizer
from rlvr_world.world import State

GOLD = "r2 c3 door open lamp off gem yes"
PERFECT = "r2 c3 door open lamp off gem yes"
WRONG_LAMP = "r2 c3 door open lamp on gem yes"
WRONG_CELL = "r5 c6 door open lamp off gem yes"
GARBAGE = "door open lamp off"


def test_parse_state_is_order_and_separator_tolerant() -> None:
    expected = State(2, 3, "open", "off", "yes")
    for text in (
        GOLD,
        "r2 c3 door=open lamp=off gem=yes",
        "gem yes lamp off door open c3 r2",
        "  R2   C3   DOOR open   LAMP off  GEM yes  ",
        "r 2 c 3 door: open lamp: off gem: yes",
        GOLD + " <eos>",
    ):
        assert parse_state(text) == expected
    assert parse_state(GARBAGE) is None
    assert parse_state("r2 c3 door open lamp off gem maybe") is None


def test_parse_state_rejects_token_ids() -> None:
    with pytest.raises(TypeError):
        parse_state([1, 2, 3])  # type: ignore[arg-type]


#: One wrong value in each of the five fields, everything else correct.
SINGLE_FIELD_ERRORS = {
    "row": "r6 c3 door open lamp off gem yes",
    "col": "r2 c7 door open lamp off gem yes",
    "door": "r2 c3 door shut lamp off gem yes",
    "lamp": WRONG_LAMP,
    "gem": "r2 c3 door open lamp off gem no",
}


def test_exact_match_reward_hand_computed() -> None:
    assert exact_match_reward(PERFECT, GOLD) == 1.0
    assert exact_match_reward(WRONG_LAMP, GOLD) == 0.0
    assert exact_match_reward(WRONG_CELL, GOLD) == 0.0
    assert exact_match_reward(GARBAGE, GOLD) == 0.0
    # A perfect prediction is maximal; every single-field error is strictly lower.
    for field, broken in SINGLE_FIELD_ERRORS.items():
        assert exact_match_reward(broken, GOLD) == 0.0, field


def test_token_f1_reward_hand_computed() -> None:
    # The canonical rendering has 8 tokens.
    assert canonical_tokens(GOLD) is not None
    assert len(canonical_tokens(GOLD) or []) == 8
    assert token_f1_reward(PERFECT, GOLD) == 1.0
    # One wrong field: 7 of 8 tokens shared -> precision = recall = 7/8.
    assert token_f1_reward(WRONG_LAMP, GOLD) == pytest.approx(7 / 8)
    # Row and column both wrong: 6 of 8 shared.
    assert token_f1_reward(WRONG_CELL, GOLD) == pytest.approx(6 / 8)
    # Nothing in common.
    assert token_f1_reward("r9 c9 shut on no", GOLD) == 0.0
    assert token_f1_reward(GARBAGE, GOLD) == 0.0


def test_structural_reward_hand_computed() -> None:
    assert structural_reward(PERFECT, GOLD) == 1.0
    for field, broken in SINGLE_FIELD_ERRORS.items():
        assert structural_reward(broken, GOLD) == pytest.approx(4 / 5), field
    assert structural_reward(WRONG_CELL, GOLD) == pytest.approx(3 / 5)
    assert structural_reward("r9 c9 door shut lamp on gem no", GOLD) == 0.0
    assert structural_reward(GARBAGE, GOLD) == 0.0


def test_perfect_prediction_is_maximal_and_wrong_is_lower() -> None:
    for scheme in REWARD_SCHEMES:
        perfect = reward_from_text(PERFECT, GOLD, scheme)
        assert perfect == 1.0
        for field, broken in SINGLE_FIELD_ERRORS.items():
            assert perfect > reward_from_text(broken, GOLD, scheme), (scheme, field)
        assert perfect > reward_from_text(WRONG_CELL, GOLD, scheme)
        assert perfect > reward_from_text(GARBAGE, GOLD, scheme)
        assert 0.0 <= reward_from_text(GARBAGE, GOLD, scheme) <= 1.0


def test_reward_is_invariant_to_tokenisation() -> None:
    """The headline invariant: decode first, then score."""
    tok = Tokenizer()
    for prediction in (PERFECT, WRONG_LAMP, WRONG_CELL, GARBAGE, "shut on no"):
        merged = tok.encode(prediction)
        chars = tok.encode_charwise(prediction)
        assert merged != chars  # the two tokenisations really are different
        assert tok.decode(merged) == tok.decode(chars) == prediction
        for scheme in REWARD_SCHEMES:
            assert reward_from_tokens(tok, merged, GOLD, scheme) == reward_from_tokens(
                tok, chars, GOLD, scheme
            )


def test_reward_is_invariant_to_field_order_and_spacing() -> None:
    tok = Tokenizer()
    reordered = "gem yes lamp off door open c3 r2"
    spaced = "r2  c3   door=open  lamp=off  gem=yes"
    for scheme in REWARD_SCHEMES:
        base = reward_from_text(GOLD, GOLD, scheme)
        assert reward_from_text(reordered, GOLD, scheme) == base
        assert reward_from_text(spaced, GOLD, scheme) == base
        assert reward_from_tokens(
            tok, tok.encode(reordered), GOLD, scheme
        ) == reward_from_tokens(tok, tok.encode(GOLD), GOLD, scheme)


def test_canonical_tokens_and_score_rollouts() -> None:
    tok = Tokenizer()
    assert canonical_tokens(GOLD) == GOLD.split(" ")
    assert canonical_tokens(GARBAGE) is None
    rollouts = [tok.encode(p) for p in (PERFECT, WRONG_LAMP, GARBAGE)]
    scores = score_rollouts(tok, rollouts, [GOLD] * 3, "binary")
    assert scores == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        score_rollouts(tok, rollouts, [GOLD], "binary")


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        reward_from_text(PERFECT, GOLD, "perceptual")
