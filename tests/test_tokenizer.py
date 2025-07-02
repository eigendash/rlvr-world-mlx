"""The tokeniser must round-trip, and must offer two tokenisations of one string."""

from __future__ import annotations

import pytest

from rlvr_world.tokenizer import (
    ALPHABET,
    BOS,
    EOS,
    PAD,
    SEP,
    SPECIALS,
    Tokenizer,
)


STATE = "r3 c5 door open lamp off gem yes"
QUESTION = f"{STATE} act east"


def test_round_trip_both_tokenisations() -> None:
    tok = Tokenizer()
    for text in (STATE, QUESTION, "r0 c0 door shut lamp on gem no", "act west"):
        assert tok.decode(tok.encode(text)) == text
        assert tok.decode(tok.encode_charwise(text)) == text


def test_two_tokenisations_differ_but_decode_alike() -> None:
    tok = Tokenizer()
    merged = tok.encode(STATE)
    chars = tok.encode_charwise(STATE)
    assert len(merged) < len(chars)
    assert merged != chars
    assert tok.decode(merged) == tok.decode(chars) == STATE

    # Keyword pieces really are used, and characters really are spelled out.
    assert any(tok.itos[i] == "door" for i in merged)
    assert [tok.itos[i] for i in chars[:4]] == ["r", "3", " ", "c"]


def test_greedy_match_prefers_longer_keywords() -> None:
    tok = Tokenizer()
    ids = tok.encode("north on no")
    assert [tok.itos[i] for i in ids] == ["north", " ", "on", " ", "no"]


def test_specials_are_stripped_by_default() -> None:
    tok = Tokenizer()
    prompt = tok.encode_prompt(STATE)
    assert tok.itos[prompt[0]] == BOS
    assert tok.itos[prompt[-1]] == SEP
    assert tok.decode(prompt) == STATE
    assert tok.decode(prompt, keep_specials=True).startswith(BOS)

    response = tok.encode_response(STATE)
    assert tok.itos[response[-1]] == EOS
    assert tok.decode(response) == STATE


def test_padding_is_dropped_and_ids_are_stable() -> None:
    tok = Tokenizer()
    assert tok.itos[tok.pad_id] == PAD
    assert tok.decode([tok.pad_id, *tok.encode("gem yes"), tok.pad_id]) == "gem yes"
    # Every piece of the vocabulary is unique.
    assert len(set(tok.itos)) == len(tok.itos)
    assert set(SPECIALS).issubset(set(tok.itos))
    assert set(ALPHABET).issubset(set(tok.itos))


def test_unknown_character_is_rejected() -> None:
    tok = Tokenizer()
    with pytest.raises(ValueError):
        tok.encode("door @ open")
    with pytest.raises(ValueError):
        tok.encode_charwise("lamp §")
