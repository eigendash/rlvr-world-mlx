"""Model contracts, causality, and the MLE objective against a reference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from rlvr_world.model import ModelConfig, WorldModel, token_accuracy
from rlvr_world.mle import (
    build_example,
    example_batch,
    masked_sequence_mean,
    mle_loss,
    nll_per_sequence,
    sequence_logprobs,
    train_mle,
)
from rlvr_world.tokenizer import Tokenizer
from rlvr_world.world import GridWorld, split_transitions

VOCAB = 16
CFG = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=32)


def world_model_config(tokenizer: Tokenizer) -> ModelConfig:
    """A tiny model sized for the real tokeniser's vocabulary."""
    return ModelConfig(
        vocab_size=len(tokenizer), d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64
    )


@pytest.fixture
def model() -> WorldModel:
    mx.random.seed(0)
    return WorldModel(CFG)


def test_logits_shape_and_dtype(model: WorldModel) -> None:
    ids = mx.random.randint(0, VOCAB, (3, 7))
    logits = model(ids)
    mx.eval(logits)
    assert logits.shape == (3, 7, VOCAB)
    assert logits.dtype == mx.float32

    with pytest.raises(ValueError):
        model(mx.zeros((4,), dtype=mx.int32))
    with pytest.raises(ValueError):
        model(mx.zeros((1, CFG.max_len + 1), dtype=mx.int32))


def test_logprobs_are_normalised(model: WorldModel) -> None:
    ids = mx.random.randint(0, VOCAB, (2, 5))
    total = mx.exp(model.logprobs(ids)).sum(axis=-1)
    mx.eval(total)
    assert np.allclose(np.array(total), 1.0, atol=1e-5)


def test_attention_is_causal(model: WorldModel) -> None:
    """Changing a later token must not change earlier logits."""
    ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=mx.int32)
    changed = mx.array([[1, 2, 3, 11, 12], [6, 7, 8, 13, 14]], dtype=mx.int32)
    a = np.array(model(ids))
    b = np.array(model(changed))
    assert np.allclose(a[:, :3, :], b[:, :3, :], atol=1e-5)
    # The final position genuinely depends on the changed value.
    assert not np.allclose(a[:, 3:, :], b[:, 3:, :], atol=1e-6)


def test_mle_loss_matches_cross_entropy_reference(model: WorldModel) -> None:
    ids = mx.random.randint(0, VOCAB, (4, 6))
    mask = mx.array(np.tile([0, 0, 1, 1, 1, 1], (4, 1)), dtype=mx.float32)
    mine = float(mle_loss(model, ids, mask))

    logits = model(ids)[:, :-1, :]
    targets = ids[:, 1:]
    reference = mx.fast.cross_entropy(logits, targets)
    expected = float((reference * mask[:, 1:]).sum() / mask[:, 1:].sum())
    assert mine == pytest.approx(expected, rel=1e-5)


def test_mle_loss_is_log_vocab_for_a_uniform_model() -> None:
    class Uniform(WorldModel):
        def __call__(self, ids: mx.array) -> mx.array:  # type: ignore[override]
            batch, length = ids.shape
            return mx.zeros((batch, length, self.cfg.vocab_size))

    uniform = Uniform(CFG)
    ids = mx.random.randint(0, VOCAB, (2, 5))
    mask = mx.ones((2, 5))
    assert float(mle_loss(uniform, ids, mask)) == pytest.approx(np.log(VOCAB), rel=1e-5)


def test_loss_mask_is_respected(model: WorldModel) -> None:
    """Padded / prompt positions must not contribute to the loss."""
    ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    full = mx.ones((1, 5))
    partial = mx.array([[0.0, 0.0, 1.0, 1.0, 1.0]])
    # The loss over a subset of positions equals the mean over exactly those.
    logp = np.array(sequence_logprobs(model, ids))[0]
    expected = -float(np.mean([logp[1], logp[2], logp[3]]))
    assert float(mle_loss(model, ids, partial)) == pytest.approx(expected, rel=1e-5)
    assert float(mle_loss(model, ids, partial)) != float(mle_loss(model, ids, full))


def test_masked_sequence_mean_shapes(model: WorldModel) -> None:
    values = mx.array([[1.0, 2.0, 3.0, 9.0], [4.0, 9.0, 9.0, 9.0]])
    mask = mx.array([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    got = np.array(masked_sequence_mean(values, mask))
    assert got == pytest.approx([2.0, 4.0])

    ids = mx.random.randint(0, VOCAB, (2, 6))
    nll = nll_per_sequence(model, ids, mx.ones((2, 6)))
    acc = token_accuracy(model(ids)[:, :-1, :], ids[:, 1:], mx.ones((2, 5)))
    mx.eval(nll, acc)
    assert nll.shape == (2,) and float(nll.sum()) > 0.0
    assert 0.0 <= float(acc) <= 1.0


def test_example_batch_has_a_masked_prompt() -> None:
    tokenizer = Tokenizer()
    world = GridWorld(rows=4, cols=4, seed=0)
    transitions, _ = split_transitions(world.transitions(), eval_frac=0.25, seed=0)
    ids, mask = example_batch(tokenizer, transitions[:3])
    prompt, response = build_example(tokenizer, transitions[0])
    ids_np, mask_np = np.array(ids), np.array(mask)
    assert ids_np.shape[0] == 3
    assert mask_np.sum(axis=1).tolist() == [len(response)] * 3
    assert mask_np[0, : len(prompt)].sum() == 0.0
    # The prompt is exactly the encoded question.
    assert ids_np[0, : len(prompt)].tolist() == prompt
    assert tokenizer.decode(ids_np[0, len(prompt) :].tolist()) == transitions[0].answer


def test_mle_gradients_are_finite() -> None:
    tokenizer = Tokenizer()
    world = GridWorld(rows=4, cols=4, seed=1)
    transitions, _ = split_transitions(world.transitions(), eval_frac=0.2, seed=0)
    ids, mask = example_batch(tokenizer, transitions[:8])
    mx.random.seed(3)
    model = WorldModel(world_model_config(tokenizer))
    _, grads = mx.value_and_grad(mle_loss)(model, ids, mask)
    leaves = tree_flatten(grads)
    assert leaves
    for _, value in leaves:
        assert bool(mx.all(mx.isfinite(value)))


def test_mle_training_reduces_the_loss_on_a_tiny_task() -> None:
    tokenizer = Tokenizer()
    world = GridWorld(rows=4, cols=4, seed=2)
    train, eval_ = split_transitions(world.transitions(), eval_frac=0.2, seed=0)
    mx.random.seed(1)
    model = WorldModel(world_model_config(tokenizer))
    ids, mask = example_batch(tokenizer, train[:16])
    before = float(mle_loss(model, ids, mask))
    train_mle(model, tokenizer, train[:16], steps=60, batch_size=8, lr=5e-3, log_every=1000)
    after = float(mle_loss(model, ids, mask))
    assert after < before
