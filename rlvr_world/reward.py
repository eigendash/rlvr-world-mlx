"""Verifiable rewards, computed on the *decoded* next-state prediction.

This is the part of RLVR-World that the paper stresses: the world model emits
token sequences, but the reward is a metric on the decoded prediction,

    R_i = sign(D) * D(decode(o_i), s'),

with ``sign(D) = +1`` for metrics that are better when larger.  Nothing here
ever looks at token ids: every reward first decodes the prediction to text and
then extracts the state from that text, so two tokenisations (or two orderings
of the fields) that decode to the same state get the same reward.

Three metrics are provided, all in ``[0, 1]``:

``binary``
    Exact match of the extracted state, the paper's binary reward for text
    game state prediction: 1 only if the prediction is completely correct.
``token_f1``
    Token-level F1 between the canonical renderings, closer to the F1 reward
    the paper uses for web page state prediction.
``structural``
    Fraction of the five state fields (row, col, door, lamp, gem) that match --
    a dense structural distance, ``1 - d``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from .tokenizer import Tokenizer
from .world import GEM_VALUES, LAMP_VALUES, DOOR_VALUES, State

REWARD_SCHEMES: tuple[str, ...] = ("binary", "token_f1", "structural")

_ROW_RE = re.compile(r"\br\s*(\d+)")
_COL_RE = re.compile(r"\bc\s*(\d+)")
_DOOR_RE = re.compile(r"\bdoor\s*[=:]?\s*(open|shut)\b")
_LAMP_RE = re.compile(r"\blamp\s*[=:]?\s*(on|off)\b")
_GEM_RE = re.compile(r"\bgem\s*[=:]?\s*(yes|no)\b")


def parse_state(text: str) -> State | None:
    """Extract a :class:`State` from decoded text, or ``None`` if that fails.

    The extractor is the toy analogue of the paper's rule-based extractor: it
    scans for each field independently, so the field order, the whitespace and
    a ``=`` between field and value do not matter.
    """
    if not isinstance(text, str):
        raise TypeError("parse_state expects the decoded text, not token ids")
    low = text.lower()
    row = _ROW_RE.search(low)
    col = _COL_RE.search(low)
    door = _DOOR_RE.search(low)
    lamp = _LAMP_RE.search(low)
    gem = _GEM_RE.search(low)
    if not (row and col and door and lamp and gem):
        return None
    return State(
        row=int(row.group(1)),
        col=int(col.group(1)),
        door=door.group(1),
        lamp=lamp.group(1),
        gem=gem.group(1),
    )


def canonical_tokens(text: str) -> list[str] | None:
    """The field tokens of the canonical rendering, or ``None`` if unparseable."""
    state = parse_state(text)
    if state is None:
        return None
    return state.render().split(" ")


# --------------------------------------------------------------------- metrics
def exact_match_reward(prediction: str, reference: str) -> float:
    """1.0 for a completely correct decoded state, else 0.0."""
    pred = parse_state(prediction)
    gold = parse_state(reference)
    if pred is None or gold is None:
        return 0.0
    return 1.0 if pred == gold else 0.0


def token_f1_reward(prediction: str, reference: str) -> float:
    """Multiset token-level F1 between the canonical renderings."""
    pred_tokens = canonical_tokens(prediction)
    gold_tokens = canonical_tokens(reference)
    if pred_tokens is None or gold_tokens is None:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def structural_reward(prediction: str, reference: str) -> float:
    """Fraction of the five state fields that match (1 - normalised distance)."""
    pred = parse_state(prediction)
    gold = parse_state(reference)
    if pred is None or gold is None:
        return 0.0
    matches = sum(
        int(a == b)
        for a, b in zip(
            (pred.row, pred.col, pred.door, pred.lamp, pred.gem),
            (gold.row, gold.col, gold.door, gold.lamp, gold.gem),
        )
    )
    return matches / 5.0


_METRICS = {
    "binary": exact_match_reward,
    "token_f1": token_f1_reward,
    "structural": structural_reward,
}


def reward_from_text(prediction: str, reference: str, scheme: str = "binary") -> float:
    """Score one decoded prediction against the ground-truth next state."""
    try:
        metric = _METRICS[scheme]
    except KeyError:
        raise ValueError(
            f"unknown reward scheme {scheme!r}; expected one of {REWARD_SCHEMES}"
        ) from None
    return float(metric(prediction, reference))


def reward_from_tokens(
    tokenizer: Tokenizer,
    prediction_ids: Iterable[int],
    reference: str,
    scheme: str = "binary",
) -> float:
    """Decode first, then score -- the order the paper insists on."""
    return reward_from_text(tokenizer.decode(prediction_ids), reference, scheme)


def score_rollouts(
    tokenizer: Tokenizer,
    rollouts: Sequence[Sequence[int]],
    references: Sequence[str],
    scheme: str = "binary",
) -> list[float]:
    """Score a flat list of rollouts against a per-rollout reference string."""
    if len(rollouts) != len(references):
        raise ValueError("rollouts and references must have the same length")
    return [
        reward_from_tokens(tokenizer, ids, ref, scheme)
        for ids, ref in zip(rollouts, references)
    ]


def score_texts(
    predictions: Sequence[str],
    references: Sequence[str],
    scheme: str = "binary",
) -> list[float]:
    """Score already-decoded predictions against reference strings."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    return [
        reward_from_text(pred, ref, scheme)
        for pred, ref in zip(predictions, references)
    ]


def field_values() -> dict[str, tuple[str, ...]]:
    """The legal values of each state field, for tests and analysis."""
    return {"door": DOOR_VALUES, "lamp": LAMP_VALUES, "gem": GEM_VALUES}
