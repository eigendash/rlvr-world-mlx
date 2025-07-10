"""The world must be deterministic, Markov in the rendered state, and cleanly split."""

from __future__ import annotations

import pytest

from rlvr_world.world import (
    ACTIONS,
    MOVES,
    GridWorld,
    State,
    destination_coverage,
    split_transitions,
    unchanged_fraction,
)


def test_initial_state_is_the_top_left_free_cell() -> None:
    world = GridWorld(rows=6, cols=6, seed=1)
    state = world.initial_state()
    assert (state.row, state.col) == (0, 0)
    assert state == world.state_at(0, 0)
    assert world.is_free(0, 0)


def test_transitions_are_deterministic_and_inside_the_grid() -> None:
    world = GridWorld(rows=7, cols=5, seed=2)
    for transition in world.transitions():
        assert transition.next_state == world.step(transition.state, transition.action)
        # Repeating the step from the same state gives the same answer.
        assert world.step(transition.state, transition.action) == transition.next_state
        assert 0 <= transition.next_state.row < world.rows
        assert 0 <= transition.next_state.col < world.cols
        assert world.is_free(transition.next_state.row, transition.next_state.col)


def test_blocked_moves_leave_the_state_alone() -> None:
    world = GridWorld(rows=4, cols=4, seed=3)
    # (0, 0) is free; moving north or west leaves the grid.
    state = world.state_at(0, 0)
    assert world.step(state, "north") == state
    assert world.step(state, "west") == state
    # A wall cell is never entered.
    world.walls[0, 1] = True
    assert world.step(state, "east") == state
    world.walls[0, 1] = False
    moved = world.step(state, "east")
    assert (moved.row, moved.col) == (0, 1)
    assert moved != state


def test_state_properties_always_match_the_cell_table() -> None:
    """The rendered state is a pure function of the cell, as the task assumes."""
    world = GridWorld(rows=6, cols=6, seed=4)
    for row, col in world.free_cells():
        state = world.state_at(row, col)
        props = world.props(row, col)
        assert (state.door, state.lamp, state.gem) == (
            props["door"],
            props["lamp"],
            props["gem"],
        )
        # Markov property: the same (row, col) reached from anywhere behaves alike.
        rebuilt = State(row, col, props["door"], props["lamp"], props["gem"])
        for action in ACTIONS:
            assert world.step(state, action) == world.step(rebuilt, action)


def test_unknown_action_is_rejected() -> None:
    world = GridWorld(rows=4, cols=4, seed=5)
    with pytest.raises(ValueError):
        world.step(world.initial_state(), "up")


def test_rendering_round_trips_through_the_grid() -> None:
    world = GridWorld(rows=5, cols=5, seed=6)
    for row, col in world.free_cells():
        text = world.state_at(row, col).render()
        assert f"r{row} c{col}" in text
        for value in world.props(row, col).values():
            assert value in text


def test_split_is_disjoint_covering_and_reproducible() -> None:
    world = GridWorld(rows=8, cols=8, seed=7)
    transitions = world.transitions()
    train_a, eval_a = split_transitions(transitions, eval_frac=0.25, seed=11)
    train_b, eval_b = split_transitions(transitions, eval_frac=0.25, seed=11)
    assert train_a == train_b and eval_a == eval_b

    # No example is in both halves, and together they are the whole set.
    keys_a = {(t.state, t.action) for t in train_a}
    keys_e = {(t.state, t.action) for t in eval_a}
    assert not (keys_a & keys_e)
    assert keys_a | keys_e == {(t.state, t.action) for t in transitions}
    assert len(eval_a) == round(len(transitions) * 0.25)


def test_destination_coverage_and_unchanged_fraction() -> None:
    world = GridWorld(rows=8, cols=8, seed=8)
    transitions = world.transitions()
    train, eval_ = split_transitions(transitions, eval_frac=0.2, seed=0)
    coverage = destination_coverage(train, eval_)
    assert 0.0 <= coverage <= 1.0
    # The grid is small, so held-out destinations are nearly always seen in train.
    assert coverage > 0.8

    unchanged = unchanged_fraction(transitions)
    assert 0.0 < unchanged < 1.0
    # (0, 0) has two out-of-bounds moves unless its neighbours are walls.
    assert unchanged_fraction([t for t in transitions if t.state.row == 0]) > 0.0


def test_action_set_matches_move_table() -> None:
    assert set(ACTIONS) == set(MOVES)
    for dr, dc in MOVES.values():
        assert abs(dr) + abs(dc) == 1
