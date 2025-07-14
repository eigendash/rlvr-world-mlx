"""The RLVR stage: GRPO on decoded prediction metrics, with a KL anchor.

The paper's objective (Eq. 1) is the clipped GRPO surrogate with a KL penalty
to the reference model::

    J(theta) = E[ (1/G) sum_i (1/|o_i|) sum_t ( min(r_i,t * A_i, clip(r_i,t) * A_i)
                                                  - beta * D_KL[p_theta || p_ref]) ]

with the group-relative advantage ``A_i = (R_i - mean(R)) / std(R)`` (Eq. 1,
Section 3) and the reward ``R_i = sign(D) * D(decode(o_i), s')`` (Eq. 3) computed
on the *decoded* prediction by :mod:`rlvr_world.reward`.

Two choices are ours and are called out in the README:

* the KL is the per-token ``k3`` estimator used by GRPO implementations,
  ``exp(log p_ref - log p_theta) - (log p_ref - log p_theta) - 1``, whose mean
  under the policy is exactly ``D_KL[p_theta || p_ref]`` (verified in the tests
  against the full-vocabulary KL);
* the RL term is multiplied by an explicit coefficient ``rl_coef`` and there is
  an optional MLE anchor term, so that the limit ``rl_coef = 0`` can be checked.
  The paper's RLVR stage is ``rl_coef = 1, mle_coef = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .mle import masked_sequence_mean, mle_loss, sequence_logprobs
from .model import WorldModel
from .rollout import group_rewards, sample_group
from .tokenizer import Tokenizer
from .world import Transition


@dataclass
class RLVRConfig:
    """Everything that controls the RLVR stage."""

    rl_coef: float = 1.0
    kl_coef: float = 1e-3  # the paper's KL loss coefficient for the LLM tasks
    mle_coef: float = 0.0  # ours: an optional supervised anchor (off by default)
    clip_eps: float = 0.2
    adv_eps: float = 1e-6
    temperature: float = 1.0
    max_new_tokens: int = 20


# --------------------------------------------------------------- advantages
def group_advantages(rewards: mx.array, adv_eps: float = 1e-6) -> mx.array:
    """``(B, G) -> (B, G)`` group-normalised advantages.

    Uses the unbiased standard deviation (``ddof=1``), as the reference GRPO
    implementations do.  A group whose rewards are all equal carries no signal,
    so its advantages are set to zero instead of dividing by ~0.
    """
    array = np.array(rewards, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("rewards must have shape (batch, group)")
    if array.shape[1] < 2:
        return mx.zeros_like(rewards)
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, ddof=1, keepdims=True)
    safe = np.where(std < adv_eps, 1.0, std)
    advantages = np.where(std < adv_eps, 0.0, (array - mean) / safe)
    return mx.array(advantages.astype(np.float32))


# ------------------------------------------------------------------ surrogate
def clipped_surrogate(ratio: mx.array, advantages: mx.array, clip_eps: float) -> mx.array:
    """``min(r * A, clip(r, 1-eps, 1+eps) * A)``, elementwise."""
    clipped = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return mx.minimum(ratio * advantages, clipped * advantages)


def clip_fraction(ratio: mx.array, mask: mx.array, clip_eps: float) -> float:
    outside = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).astype(mx.float32)
    return float((outside * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0)))


# ------------------------------------------------------------------- KL terms
def kl_k3(logp_policy: mx.array, logp_reference: mx.array) -> mx.array:
    """Per-token KL estimator, zero when the two distributions agree."""
    diff = logp_reference - logp_policy
    return mx.exp(diff) - diff - 1.0


def kl_exact(policy_logits: mx.array, reference_logits: mx.array) -> mx.array:
    """Exact ``D_KL(p_theta || p_ref)`` per position, over the full vocabulary."""
    logp = nn.log_softmax(policy_logits, axis=-1)
    logq = nn.log_softmax(reference_logits, axis=-1)
    return (mx.exp(logp) * (logp - logq)).sum(axis=-1)


# ---------------------------------------------------------------------- loss
def rlvr_loss(
    model: WorldModel,
    reference: WorldModel,
    input_ids: mx.array,
    loss_mask: mx.array,
    advantages: mx.array,
    old_logp: mx.array,
    cfg: RLVRConfig,
    mle_batch: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    """The full RLVR objective, returned with a small diagnostics dict.

    ``input_ids`` are rollouts (prompt + generated response, right-padded),
    ``loss_mask`` is 1 on the response tokens, ``old_logp`` is the behaviour
    policy's log-probability of those tokens (``(B, T-1)``, treated as a
    constant), and ``advantages`` is one scalar per sequence.
    """
    if cfg.mle_coef != 0.0 and mle_batch is None:
        raise ValueError("mle_coef is non-zero but no mle_batch was given")

    logp = sequence_logprobs(model, input_ids)
    logp_ref = mx.stop_gradient(sequence_logprobs(reference, input_ids))
    mask = loss_mask[:, 1:]

    ratio = mx.exp(logp - old_logp)
    surrogate_tok = clipped_surrogate(ratio, advantages[:, None], cfg.clip_eps)
    kl_tok = kl_k3(logp, logp_ref)

    surrogate = masked_sequence_mean(surrogate_tok, mask).mean()
    kl = masked_sequence_mean(kl_tok, mask).mean()

    rl_term = -(surrogate - cfg.kl_coef * kl)
    loss = cfg.rl_coef * rl_term
    if cfg.mle_coef != 0.0:
        assert mle_batch is not None
        loss = loss + cfg.mle_coef * mle_loss(model, *mle_batch)

    aux = {
        "surrogate": mx.stop_gradient(surrogate),
        "kl": mx.stop_gradient(kl),
        "rl_term": mx.stop_gradient(rl_term),
        "clip_fraction": mx.stop_gradient(
            mx.array(clip_fraction(ratio, mask, cfg.clip_eps))
        ),
    }
    return loss, aux


# ------------------------------------------------------------------- training
@dataclass
class RLVRHistory:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    reward_std: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    signal: list[float] = field(default_factory=list)

    def record(self, step: int, loss: float, reward: float, reward_std: float, kl: float, signal: float) -> None:
        self.steps.append(step)
        self.loss.append(loss)
        self.reward.append(reward)
        self.reward_std.append(reward_std)
        self.kl.append(kl)
        self.signal.append(signal)


def train_rlvr(
    model: WorldModel,
    reference: WorldModel,
    tokenizer: Tokenizer,
    transitions: Sequence[Transition],
    steps: int = 100,
    batch_size: int = 8,
    group_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    seed: int = 0,
    cfg: RLVRConfig | None = None,
    scheme: str = "binary",
    log_every: int = 25,
    log: Callable[[str], None] = print,
) -> RLVRHistory:
    """Post-train ``model`` in place with GRPO against decoded rewards."""
    cfg = cfg or RLVRConfig()
    if not transitions:
        raise ValueError("no training transitions")
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    history = RLVRHistory()

    prompt_len = len(tokenizer.encode_prompt(transitions[0].question))

    def objective(m: WorldModel) -> mx.array:
        return rlvr_loss(
            m, reference, sequences, mask, advantages.reshape(-1), old_logp, cfg
        )[0]

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(transitions), size=min(batch_size, len(transitions)))
        batch = [transitions[int(i)] for i in idx]
        prompts = mx.array(
            np.stack([tokenizer.encode_prompt(t.question) for t in batch]).astype(np.int32)
        )
        seed_key = mx.random.key(seed * 100_003 + step)
        sequences, mask = sample_group(
            model,
            tokenizer,
            prompts,
            group_size,
            cfg.max_new_tokens,
            seed_key,
            temperature=cfg.temperature,
        )
        rewards = group_rewards(
            tokenizer,
            sequences,
            prompt_len,
            [t.answer for t in batch],
            group_size,
            scheme,
        )
        mx.eval(rewards)
        advantages = group_advantages(rewards.reshape(len(batch), group_size), cfg.adv_eps)
        old_logp = mx.stop_gradient(sequence_logprobs(model, sequences))

        loss_value, grads = mx.value_and_grad(objective)(model)
        report = step % log_every == 0 or step == 1 or step == steps
        aux: dict[str, mx.array] = {}
        if report:
            _, aux = rlvr_loss(
                model, reference, sequences, mask, advantages.reshape(-1), old_logp, cfg
            )
            mx.eval(aux["kl"])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        reward_np = np.array(rewards.reshape(len(batch), group_size), dtype=np.float64)
        # Fraction of groups that carry any learning signal at all.
        signal = float(np.mean(reward_np.std(axis=1, ddof=1) > cfg.adv_eps))
        if report:
            history.record(
                step,
                float(loss_value),
                float(reward_np.mean()),
                float(reward_np.std(axis=1, ddof=1).mean()),
                float(aux["kl"]),
                signal,
            )
            log(
                f"  rlvr step {step:5d}  loss {float(loss_value):+.5f}"
                f"  reward {reward_np.mean():.4f}  kl {float(aux['kl']):.3e}"
                f"  groups-with-signal {signal:.2f}"
            )
    return history
