"""The toy text world: a small deterministic grid navigation task.

A state is the cell the agent occupies together with that cell's three binary
properties, rendered as one short string::

    r3 c5 door open lamp off gem yes

An action is a compass move.  The transition is deterministic: the agent moves
one cell if the destination is inside the grid and not a wall, and stays put
otherwise (that is the toy version of the paper's "unchanged cases").  The
properties of the new cell are read off the world, so the state string carries
everything needed to predict the next one -- the process is first-order Markov
once the state is written out, exactly as in the paper's sequence formulation.

The cell property tables are drawn once from a seed and never change, so the
held-out transitions are predictable from the training ones (the model has to
learn where each cell's properties live, not memorise the held-out answers).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

ACTIONS: tuple[str, ...] = ("north", "south", "east", "west")

MOVES: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}

DOOR_VALUES: tuple[str, ...] = ("open", "shut")
LAMP_VALUES: tuple[str, ...] = ("on", "off")
GEM_VALUES: tuple[str, ...] = ("yes", "no")


@dataclass(frozen=True)
class State:
    """The agent's cell plus the properties of that cell."""

    row: int
    col: int
    door: str
    lamp: str
    gem: str

    def render(self) -> str:
        return (
            f"r{self.row} c{self.col} door {self.door} "
            f"lamp {self.lamp} gem {self.gem}"
        )


@dataclass(frozen=True)
class Transition:
    """One ``(state, action) -> next state`` example."""

    state: State
    action: str
    next_state: State

    @property
    def question(self) -> str:
        return f"{self.state.render()} act {self.action}"

    @property
    def answer(self) -> str:
        return self.next_state.render()


class GridWorld:
    """A ``rows x cols`` grid with wall cells and per-cell properties."""

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        seed: int = 0,
        wall_frac: float = 0.12,
    ) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("the grid must be at least 2x2")
        self.rows = rows
        self.cols = cols
        rng = np.random.default_rng(seed)
        self.walls = rng.random((rows, cols)) < wall_frac
        # The agent starts at (0, 0), so that cell must be free.
        self.walls[0, 0] = False
        self.door = rng.choice(np.array(DOOR_VALUES), size=(rows, cols))
        self.lamp = rng.choice(np.array(LAMP_VALUES), size=(rows, cols))
        self.gem = rng.choice(np.array(GEM_VALUES), size=(rows, cols))

    # ------------------------------------------------------------------ basics
    def is_free(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and not bool(self.walls[row, col])
        )

    def props(self, row: int, col: int) -> dict[str, str]:
        return {
            "door": str(self.door[row, col]),
            "lamp": str(self.lamp[row, col]),
            "gem": str(self.gem[row, col]),
        }

    def state_at(self, row: int, col: int) -> State:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"cell {(row, col)} is outside the grid")
        props = self.props(row, col)
        return State(row, col, props["door"], props["lamp"], props["gem"])

    def initial_state(self) -> State:
        return self.state_at(0, 0)

    def free_cells(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.is_free(r, c)
        ]

    # -------------------------------------------------------------- transition
    def step(self, state: State, action: str) -> State:
        """One deterministic transition.  Blocked moves leave the state alone."""
        if action not in MOVES:
            raise ValueError(f"unknown action {action!r}")
        dr, dc = MOVES[action]
        row, col = state.row + dr, state.col + dc
        if self.is_free(row, col):
            return self.state_at(row, col)
        return state

    def transitions(self) -> list[Transition]:
        """Every ``(free cell, action)`` pair, in a deterministic order."""
        out: list[Transition] = []
        for row, col in self.free_cells():
            state = self.state_at(row, col)
            for action in ACTIONS:
                out.append(Transition(state, action, self.step(state, action)))
        return out

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions())


def split_transitions(
    transitions: Sequence[Transition],
    eval_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Transition], list[Transition]]:
    """Shuffle and split into ``(train, eval)`` with a fixed seed."""
    if not 0.0 < eval_frac < 1.0:
        raise ValueError("eval_frac must lie strictly between 0 and 1")
    items = list(transitions)
    random.Random(seed).shuffle(items)
    n_eval = max(1, round(len(items) * eval_frac))
    return items[n_eval:], items[:n_eval]


def destination_coverage(
    train: Sequence[Transition], eval_: Sequence[Transition]
) -> float:
    """Fraction of eval transitions whose destination cell is visited in train.

    This is the ceiling on what a model could possibly generalise to: a
    destination cell that never occurs in training carries properties the model
    has no way to know.
    """
    seen = {(t.next_state.row, t.next_state.col) for t in train}
    if not eval_:
        return float("nan")
    hits = sum(
        1 for t in eval_ if (t.next_state.row, t.next_state.col) in seen
    )
    return hits / len(eval_)


def unchanged_fraction(transitions: Sequence[Transition]) -> float:
    """Fraction of transitions whose next state equals the current state."""
    if not transitions:
        return float("nan")
    same = sum(1 for t in transitions if t.next_state == t.state)
    return same / len(transitions)
