"""Maximum-likelihood (next-token) training of the world model.

This is the paper's ``J_MLE`` (Eq. 2): the log-likelihood of the response given
the question, with the prompt positions masked out of the loss.  The same
module is reused later: the RLVR objective contains an optional MLE anchor
term, and with the RL coefficient at zero the whole RLVR loss has to collapse
back onto exactly this function.

Sequences are stored as a single ``(N, P + R)`` array: ``<bos> q <sep> o <eos>``
where ``q`` and ``o`` have a fixed length in this toy task.  The loss mask is 1
on the response positions (including ``<eos>``) and 0 on the question and on
padding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import WorldModel, token_accuracy
from .tokenizer import Tokenizer
from .world import Transition


def build_example(tokenizer: Tokenizer, transition: Transition) -> tuple[list[int], list[int]]:
    """``(prompt ids, response ids)`` for one transition."""
    return (
        tokenizer.encode_prompt(transition.question),
        tokenizer.encode_response(transition.answer),
    )


def pad_batch(
    sequences: Sequence[Sequence[int]], pad_id: int
) -> tuple[mx.array, mx.array]:
    """Right-pad token sequences to a rectangle, with a 1/0 mask per position."""
    if not sequences:
        raise ValueError("cannot pad an empty batch")
    width = max(len(s) for s in sequences)
    ids = np.full((len(sequences), width), pad_id, dtype=np.int32)
    mask = np.zeros((len(sequences), width), dtype=np.float32)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = seq
        mask[i, : len(seq)] = 1.0
    return mx.array(ids), mx.array(mask)


def example_batch(
    tokenizer: Tokenizer, transitions: Sequence[Transition]
) -> tuple[mx.array, mx.array]:
    """Batch of teacher-forced examples: ``(input ids, loss mask)``.

    ``input ids[i] = prompt_i + response_i``.  The loss mask is zero on the
    prompt and one on the response tokens (including ``<eos>``), so it is
    already in the alignment used with ``logits[:, :-1]`` against ``ids[:, 1:]``.
    """
    prompts: list[list[int]] = []
    responses: list[list[int]] = []
    for transition in transitions:
        prompt, response = build_example(tokenizer, transition)
        prompts.append(prompt)
        responses.append(response)
    ids, _ = pad_batch(
        [p + r for p, r in zip(prompts, responses)], tokenizer.pad_id
    )
    mask = np.zeros((len(transitions), ids.shape[1]), dtype=np.float32)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        mask[i, len(prompt) : len(prompt) + len(response)] = 1.0
    return ids, mx.array(mask)


def mle_loss(model: WorldModel, input_ids: mx.array, loss_mask: mx.array) -> mx.array:
    """Mean negative log-likelihood per response token."""
    logits = model(input_ids)[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:]
    logp = mx.take_along_axis(
        nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1
    )[..., 0]
    total = (-logp * mask).sum()
    return total / mx.maximum(mask.sum(), mx.array(1.0))


def sequence_logprobs(model: WorldModel, input_ids: mx.array) -> mx.array:
    """Log-probability of each token given its prefix, ``(B, T-1)``.

    Position ``t`` of the output is ``log p(input_ids[t + 1] | input_ids[:t+1])``.
    """
    logits = model(input_ids)[:, :-1, :]
    targets = input_ids[:, 1:]
    return mx.take_along_axis(
        nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1
    )[..., 0]


def masked_sequence_mean(values: mx.array, mask: mx.array) -> mx.array:
    """Mean of ``values`` over the mask, per sequence: ``(B, T) -> (B,)``."""
    return (values * mask).sum(axis=-1) / mx.maximum(
        mask.sum(axis=-1), mx.array(1.0)
    )


def nll_per_sequence(model: WorldModel, input_ids: mx.array, loss_mask: mx.array) -> mx.array:
    """Mean NLL of each sequence's response tokens, ``(B,)`` in nats."""
    logp = sequence_logprobs(model, input_ids)
    return masked_sequence_mean(-logp, loss_mask[:, 1:])


@dataclass
class MLEHistory:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    def record(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.loss.append(loss)


def train_mle(
    model: WorldModel,
    tokenizer: Tokenizer,
    transitions: Sequence[Transition],
    steps: int = 400,
    batch_size: int = 32,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    seed: int = 0,
    log_every: int = 100,
    log: Callable[[str], None] = print,
) -> MLEHistory:
    """Train ``model`` in place on transitions; returns the loss history."""
    if not transitions:
        raise ValueError("no training transitions")
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    def loss_fn(m: WorldModel, ids: mx.array, mask: mx.array) -> mx.array:
        return mle_loss(m, ids, mask)

    value_and_grad = mx.value_and_grad(loss_fn)
    history = MLEHistory()

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(transitions), size=min(batch_size, len(transitions)))
        batch = [transitions[int(i)] for i in idx]
        ids, mask = example_batch(tokenizer, batch)
        loss, grads = value_and_grad(model, ids, mask)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step % log_every == 0 or step == 1 or step == steps:
            history.record(step, float(loss))
            log(f"  mle step {step:5d}  nll {float(loss):.4f}")
    return history


def evaluate_mle(
    model: WorldModel,
    tokenizer: Tokenizer,
    transitions: Sequence[Transition],
    batch_size: int = 64,
) -> dict[str, float]:
    """Teacher-forced diagnostics: NLL and next-token accuracy on responses."""
    nlls: list[float] = []
    accs: list[float] = []
    for start in range(0, len(transitions), batch_size):
        batch = transitions[start : start + batch_size]
        ids, mask = example_batch(tokenizer, batch)
        logits = model(ids)[:, :-1, :]
        nll = nll_per_sequence(model, ids, mask)
        acc = token_accuracy(logits, ids[:, 1:], mask[:, 1:])
        mx.eval(nll, acc)
        nlls.extend(np.array(nll).tolist())
        accs.append(float(acc))
    return {
        "teacher_forced_nll": float(np.mean(nlls)),
        "teacher_forced_token_accuracy": float(np.mean(accs)),
    }
