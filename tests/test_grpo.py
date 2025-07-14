"""Advantages, the clipped surrogate, the KL penalty, and the RLVR limits."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from rlvr_world.grpo import (
    RLVRConfig,
    clip_fraction,
    clipped_surrogate,
    group_advantages,
    kl_exact,
    kl_k3,
    rlvr_loss,
    train_rlvr,
)
from rlvr_world.mle import example_batch, mle_loss, sequence_logprobs
from rlvr_world.model import ModelConfig, WorldModel
from rlvr_world.tokenizer import Tokenizer
from rlvr_world.world import GridWorld, split_transitions


def tiny_model(tokenizer: Tokenizer, seed: int = 0) -> WorldModel:
    mx.random.seed(seed)
    return WorldModel(
        ModelConfig(
            vocab_size=len(tokenizer), d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64
        )
    )


def clone_params(model: WorldModel):
    return tree_map(lambda a: mx.array(a), model.parameters())


def param_arrays(model: WorldModel) -> list[np.ndarray]:
    return [np.array(v) for _, v in tree_flatten(model.parameters())]


# ------------------------------------------------------------------ advantages
def test_group_advantages_match_hand_computation() -> None:
    rewards = mx.array([[1.0, 2.0, 3.0, 4.0]])
    adv = np.array(group_advantages(rewards))[0]
    # mean 2.5, unbiased std sqrt(5/3) = 1.290994...
    expected = np.array([-1.5, -0.5, 0.5, 1.5]) / np.sqrt(5 / 3)
    assert adv == pytest.approx(expected, rel=1e-6)
    assert adv.mean() == pytest.approx(0.0, abs=1e-5)
    std = np.sqrt(5 / 3)
    assert adv.std(ddof=1) == pytest.approx(1.0, rel=1e-5)


def test_group_advantages_are_per_group() -> None:
    rewards = mx.array([[0.0, 1.0], [5.0, 5.0], [2.0, 0.0]])
    adv = np.array(group_advantages(rewards))
    # Group 0: [0, 1] -> mean 0.5, unbiased std sqrt(0.5).
    assert adv[0] == pytest.approx(np.array([-0.5, 0.5]) / np.sqrt(0.5), rel=1e-6)
    # Group 1 has no spread: zero advantage rather than a division by zero.
    assert adv[1] == pytest.approx([0.0, 0.0])
    assert np.all(np.isfinite(adv))
    # Linear shift and positive scaling leave the advantages unchanged.
    shifted = np.array(group_advantages(rewards * 3.0 + 7.0))
    assert shifted == pytest.approx(adv, rel=1e-5)


def test_group_advantages_degenerate_inputs() -> None:
    single = np.array(group_advantages(mx.array([[1.0], [2.0]])))
    assert np.array_equal(single, np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        group_advantages(mx.array([1.0, 2.0]))


# ------------------------------------------------------------------- surrogate
def test_clipped_surrogate_hand_computed() -> None:
    eps = 0.2
    # Positive advantage: the pessimistic (lower) branch wins.
    assert float(clipped_surrogate(mx.array([1.5]), mx.array([1.0]), eps)[0]) == pytest.approx(1.2)
    assert float(clipped_surrogate(mx.array([0.5]), mx.array([1.0]), eps)[0]) == pytest.approx(0.5)
    # Negative advantage: the clipped branch is now the pessimistic one.
    assert float(clipped_surrogate(mx.array([1.5]), mx.array([-1.0]), eps)[0]) == pytest.approx(-1.5)
    assert float(clipped_surrogate(mx.array([0.5]), mx.array([-1.0]), eps)[0]) == pytest.approx(-0.8)
    # At ratio 1 the surrogate is exactly the advantage.
    assert float(clipped_surrogate(mx.array([1.0]), mx.array([0.7]), eps)[0]) == pytest.approx(0.7)
    # Past the clip the surrogate is capped, so a larger ratio earns nothing.
    good = clipped_surrogate(mx.array([1.1, 3.0]), mx.array([1.0, 1.0]), eps)
    assert float(good[0]) == pytest.approx(1.1)
    assert float(good[1]) == pytest.approx(1.0 + eps)

    ratio = mx.array([0.5, 1.0, 1.5])
    mask = mx.ones((1, 3))
    assert clip_fraction(ratio.reshape(1, 3), mask, eps) == pytest.approx(2 / 3)


# -------------------------------------------------------------------- KL terms
def test_kl_penalty_is_zero_when_policy_equals_reference() -> None:
    logits = mx.random.normal((2, 5, 7))
    logp = nn.log_softmax(logits, axis=-1)
    penalty = kl_k3(logp, logp)
    mx.eval(penalty)
    assert np.allclose(np.array(penalty), 0.0, atol=1e-6)
    exact = kl_exact(logits, logits)
    mx.eval(exact)
    assert np.allclose(np.array(exact), 0.0, atol=1e-6)


def test_kl_k3_expectation_equals_the_exact_kl() -> None:
    """Full enumeration over a tiny vocabulary: E_q[k3] == KL(q || p)."""
    policy_logits = mx.array([[2.0, 0.5, -1.0, 0.0]])
    reference_logits = mx.array([[0.5, 1.5, 0.5, -2.0]])
    logp = nn.log_softmax(policy_logits, axis=-1)
    logq = nn.log_softmax(reference_logits, axis=-1)
    probs = np.exp(np.array(logp))

    k3 = np.array(kl_k3(logp, logq))
    enumerated = float((probs * k3).sum())
    exact = float(np.array(kl_exact(policy_logits, reference_logits)).sum())
    assert enumerated == pytest.approx(exact, rel=1e-4, abs=1e-6)
    assert exact > 0.0


def test_kl_is_asymmetric_and_positive() -> None:
    first = mx.array([[2.0, 0.0, 0.0]])
    second = mx.array([[0.0, 1.0, 1.0]])
    forward = float(kl_exact(first, second).sum())
    backward = float(kl_exact(second, first).sum())
    assert forward > 0.0 and backward > 0.0
    assert forward != pytest.approx(backward, rel=1e-3)


# ------------------------------------------------------------------ RLVR loss
@pytest.fixture
def setup():
    tokenizer = Tokenizer()
    world = GridWorld(rows=4, cols=4, seed=0)
    train, _ = split_transitions(world.transitions(), eval_frac=0.25, seed=0)
    model = tiny_model(tokenizer)
    reference = tiny_model(tokenizer, seed=1)
    reference.update(clone_params(model))  # policy == reference to begin with
    ids, mask = example_batch(tokenizer, train[:4])
    return tokenizer, train, model, reference, ids, mask


def test_rlvr_loss_is_the_negative_advantage_when_ratio_is_one(setup) -> None:
    """With on-policy ratios and an identical reference the loss is -mean(A)."""
    tokenizer, train, model, reference, ids, mask = setup
    advantages = mx.array([0.5, -0.25, 1.0, -0.75])  # one per sequence
    old_logp = mx.stop_gradient(sequence_logprobs(model, ids))
    cfg = RLVRConfig(rl_coef=1.0, kl_coef=1e-3, mle_coef=0.0)
    loss, aux = rlvr_loss(model, reference, ids, mask, advantages, old_logp, cfg)
    mx.eval(loss, aux["kl"], aux["surrogate"])
    assert float(aux["kl"]) == pytest.approx(0.0, abs=1e-6)
    # ratio == 1 everywhere, so the surrogate is the mean advantage.
    assert float(aux["surrogate"]) == pytest.approx(float(advantages.mean()), abs=1e-5)
    assert float(loss) == pytest.approx(-0.125, abs=1e-5)
    assert float(aux["clip_fraction"]) == 0.0


def test_rlvr_loss_reduces_to_mle_when_the_rl_coefficient_is_zero(setup) -> None:
    """The clean limit: mle_coef=1, rl_coef=0, kl_coef=0 is exactly the MLE loss."""
    tokenizer, train, model, reference, ids, mask = setup
    advantages = mx.array([0.5, -0.25, 1.0, -0.75])  # one per sequence
    old_logp = mx.stop_gradient(sequence_logprobs(model, ids))

    cfg = RLVRConfig(rl_coef=0.0, kl_coef=0.0, mle_coef=1.0)
    loss, _ = rlvr_loss(model, reference, ids, mask, advantages, old_logp, cfg, (ids, mask))
    expected = mle_loss(model, ids, mask)
    mx.eval(loss, expected)
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)

    grads = mx.value_and_grad(
        lambda m: rlvr_loss(m, reference, ids, mask, advantages, old_logp, cfg, (ids, mask))[0]
    )(model)
    ref_grads = mx.value_and_grad(lambda m: mle_loss(m, ids, mask))(model)
    for (_, a), (_, b) in zip(tree_flatten(grads), tree_flatten(ref_grads)):
        assert np.allclose(np.array(a), np.array(b), atol=1e-6)


def test_rlvr_loss_is_a_no_op_without_the_rl_term(setup) -> None:
    """rl_coef=0 with no MLE anchor gives a zero loss and zero gradients."""
    tokenizer, train, model, reference, ids, mask = setup
    advantages = mx.array([0.5, -0.25, 1.0, -0.75])
    old_logp = mx.stop_gradient(sequence_logprobs(model, ids))
    cfg = RLVRConfig(rl_coef=0.0, kl_coef=1e-3, mle_coef=0.0)
    loss, _ = rlvr_loss(model, reference, ids, mask, advantages, old_logp, cfg)
    grads = mx.value_and_grad(
        lambda m: rlvr_loss(m, reference, ids, mask, advantages, old_logp, cfg)[0]
    )(model)
    mx.eval(loss)
    assert float(loss) == 0.0
    for _, value in tree_flatten(grads):
        assert np.allclose(np.array(value), 0.0, atol=0.0)


def test_rlvr_loss_needs_an_mle_batch_when_anchored(setup) -> None:
    tokenizer, train, model, reference, ids, mask = setup
    advantages = mx.zeros((ids.shape[0],))
    old_logp = mx.stop_gradient(sequence_logprobs(model, ids))
    cfg = RLVRConfig(rl_coef=1.0, mle_coef=1.0)
    with pytest.raises(ValueError):
        rlvr_loss(model, reference, ids, mask, advantages, old_logp, cfg)


def test_rlvr_loss_gradients_are_finite(setup) -> None:
    tokenizer, train, model, reference, ids, mask = setup
    advantages = mx.array([0.5, -0.25, 1.0, -0.75])  # one per sequence
    old_logp = mx.stop_gradient(sequence_logprobs(model, ids) + 0.1)  # off-policy-ish
    cfg = RLVRConfig(rl_coef=1.0, kl_coef=1e-3, mle_coef=0.0)
    loss, aux = rlvr_loss(model, reference, ids, mask, advantages, old_logp, cfg)
    grads = mx.value_and_grad(
        lambda m: rlvr_loss(m, reference, ids, mask, advantages, old_logp, cfg)[0]
    )(model)
    mx.eval(loss, aux["kl"])
    assert np.isfinite(float(loss)) and np.isfinite(float(aux["kl"]))
    for _, value in tree_flatten(grads):
        assert bool(mx.all(mx.isfinite(value)))


def test_masked_out_positions_do_not_change_the_loss(setup) -> None:
    tokenizer, train, model, reference, ids, mask = setup
    advantages = mx.array([0.5, -0.25, 1.0, -0.75])
    old_logp = mx.stop_gradient(sequence_logprobs(model, ids))
    cfg = RLVRConfig(rl_coef=1.0, kl_coef=1e-3)
    base, _ = rlvr_loss(model, reference, ids, mask, advantages, old_logp, cfg)

    # Append a padded position that the mask ignores.
    extended_ids = mx.concatenate([ids, mx.zeros((ids.shape[0], 1), dtype=mx.int32)], axis=1)
    extended_mask = mx.concatenate([mask, mx.zeros((mask.shape[0], 1))], axis=1)
    extended_logp = mx.concatenate(
        [old_logp, mx.zeros((old_logp.shape[0], 1))], axis=1
    )
    other, _ = rlvr_loss(
        model, reference, extended_ids, extended_mask, advantages, extended_logp, cfg
    )
    mx.eval(base, other)
    assert float(base) == pytest.approx(float(other), rel=1e-5)


def test_train_rlvr_with_rl_coefficient_zero_leaves_the_model_untouched() -> None:
    """The trainer-level version of the no-op limit."""
    tokenizer = Tokenizer()
    world = GridWorld(rows=4, cols=4, seed=0)
    train, _ = split_transitions(world.transitions(), eval_frac=0.25, seed=0)
    model = tiny_model(tokenizer)
    reference = tiny_model(tokenizer, seed=2)
    reference.update(clone_params(model))
    before = param_arrays(model)
    train_rlvr(
        model,
        reference,
        tokenizer,
        train,
        steps=5,
        batch_size=4,
        group_size=4,
        lr=1e-3,
        weight_decay=0.0,  # AdamW's decoupled decay would move the weights by itself
        seed=0,
        cfg=RLVRConfig(rl_coef=0.0, kl_coef=1e-3, mle_coef=0.0, max_new_tokens=6),
        log=lambda _: None,
    )
    for a, b in zip(before, param_arrays(model)):
        assert np.array_equal(a, b)


def test_train_rlvr_updates_the_model_when_rl_is_on() -> None:
    """A reference that differs from the policy gives the KL term work to do."""
    tokenizer = Tokenizer()
    world = GridWorld(rows=4, cols=4, seed=0)
    train, _ = split_transitions(world.transitions(), eval_frac=0.25, seed=0)
    model = tiny_model(tokenizer)
    reference = tiny_model(tokenizer, seed=2)  # deliberately not a copy
    before = param_arrays(model)
    history = train_rlvr(
        model,
        reference,
        tokenizer,
        train,
        steps=3,
        batch_size=4,
        group_size=4,
        lr=1e-3,
        weight_decay=0.0,
        seed=0,
        cfg=RLVRConfig(rl_coef=1.0, kl_coef=1e-3, mle_coef=0.0, max_new_tokens=6),
        log=lambda _: None,
    )
    after = param_arrays(model)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    assert history.steps[0] == 1 and history.steps[-1] == 3
    assert all(np.isfinite(history.loss))
    assert all(0.0 <= value <= 1.0 for value in history.reward)
    assert all(0.0 <= value <= 1.0 for value in history.signal)
    assert all(value >= 0.0 for value in history.kl)
