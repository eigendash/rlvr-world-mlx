"""Sampling and decoding rollouts from the world model.

The world model is asked a question and generates its own next-state token
sequence, token by token, conditioned on the tokens it has already produced.
That is what makes the RLVR stage different from teacher-forced MLE training:
the reward is computed on the response the model actually commits to, so
errors accumulate over the horizon exactly as they do at test time.

Sampling follows the paper's settings for the text-game task: temperature 1.0,
no nucleus truncation.  Structural pieces (``<pad>``, ``<bos>``, ``<sep>``,
``<unk>``) are forbidden during sampling; only text tokens and ``<eos>`` can be
drawn, so a rollout is always a decodable string.
"""

from __future__ import annotations

from typing import Callable, Sequence

import mlx.core as mx
import numpy as np

from .model import WorldModel
from .tokenizer import SPECIALS, BOS, EOS, PAD, SEP, UNK, Tokenizer


def forbidden_ids(tokenizer: Tokenizer) -> list[int]:
    """Ids that a rollout must never contain (everything special but ``<eos>``)."""
    banned = {tokenizer.stoi[PAD], tokenizer.stoi[BOS], tokenizer.stoi[SEP], tokenizer.stoi[UNK]}
    return sorted(banned)


def generate(
    model: WorldModel,
    prompt_ids: mx.array,
    max_new_tokens: int,
    tokenizer: Tokenizer,
    key: mx.array | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
) -> mx.array:
    """Autoregressively complete every prompt, returning ``(B, P + max_new)``."""
    banned = mx.array(forbidden_ids(tokenizer))
    sequences = prompt_ids
    finished = mx.zeros((prompt_ids.shape[0],), dtype=mx.bool_)
    for _ in range(max_new_tokens):
        logits = model(sequences)[:, -1, :]
        logits = mx.put_along_axis(
            logits, mx.broadcast_to(banned[None, :], (logits.shape[0], banned.size)),
            mx.array(-1e9), axis=-1,
        )
        if greedy:
            next_ids = mx.argmax(logits, axis=-1).astype(mx.int32)
        else:
            if key is None:
                raise ValueError("sampling requires a PRNG key")
            key, subkey = mx.random.split(key)
            next_ids = mx.random.categorical(logits / temperature, key=subkey)
        # Once a sequence has emitted <eos>, pad the rest so the mask is clean.
        next_ids = mx.where(finished, mx.array(tokenizer.pad_id, mx.int32), next_ids)
        finished = finished | (next_ids == tokenizer.eos_id)
        sequences = mx.concatenate([sequences, next_ids[:, None]], axis=1)
    return sequences


def repeat_prompts(prompt_ids: mx.array, group_size: int) -> mx.array:
    """``(B, P) -> (B * G, P)`` with the G copies of a prompt kept contiguous."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    return mx.repeat(prompt_ids, group_size, axis=0)


def response_mask(sequences: mx.array, prompt_len: int, pad_id: int) -> mx.array:
    """1 on generated text tokens (including ``<eos>``), 0 on the prompt and pads."""
    mask = (sequences[:, prompt_len:] != pad_id).astype(mx.float32)
    return mx.concatenate(
        [mx.zeros((sequences.shape[0], prompt_len)), mask], axis=1
    )


def sample_group(
    model: WorldModel,
    tokenizer: Tokenizer,
    prompt_ids: mx.array,
    group_size: int,
    max_new_tokens: int,
    key: mx.array,
    temperature: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Sample ``group_size`` responses per prompt: ``(sequences, loss mask)``."""
    repeated = repeat_prompts(prompt_ids, group_size)
    prompt_len = prompt_ids.shape[1]
    sequences = generate(
        model,
        repeated,
        max_new_tokens,
        tokenizer,
        key=key,
        temperature=temperature,
    )
    return sequences, response_mask(sequences, prompt_len, tokenizer.pad_id)


def decode_rollouts(
    tokenizer: Tokenizer, sequences: mx.array, prompt_len: int
) -> list[str]:
    """Decode only the generated part of each sequence."""
    return [tokenizer.decode(row[prompt_len:]) for row in np.array(sequences)]


def rollout_texts(
    model: WorldModel,
    tokenizer: Tokenizer,
    transitions: Sequence,
    max_new_tokens: int,
    greedy: bool = True,
) -> list[str]:
    """Complete a batch of questions (no grouping), for evaluation."""
    prompts = [tokenizer.encode_prompt(t.question) for t in transitions]
    ids = mx.array(np.stack(prompts).astype(np.int32))
    sequences = generate(
        model, ids, max_new_tokens, tokenizer, greedy=greedy, key=mx.random.key(0)
    )
    return decode_rollouts(tokenizer, sequences, ids.shape[1])


def group_rewards(
    tokenizer: Tokenizer,
    sequences: mx.array,
    prompt_len: int,
    references: Sequence[str],
    group_size: int,
    scheme: str,
    log: Callable[[str], None] | None = None,
) -> mx.array:
    """Reward every rollout, ``(B * G,)``, with ``references`` repeated per group."""
    from .reward import score_texts

    texts = decode_rollouts(tokenizer, sequences, prompt_len)
    refs = [ref for ref in references for _ in range(group_size)]
    if len(refs) != len(texts):
        raise ValueError("references and rollouts do not line up")
    return mx.array(score_texts(texts, refs, scheme))


# ---------------------------------------------------------------- evaluation
def greedy_response_texts(
    model: WorldModel,
    tokenizer: Tokenizer,
    transitions: Sequence,
    max_new_tokens: int,
) -> list[str]:
    """Decode the model's own response for each question, argmax token by token."""
    return rollout_texts(model, tokenizer, transitions, max_new_tokens, greedy=True)


def response_token_accuracy(
    tokenizer: Tokenizer,
    predicted: Sequence[str],
    transitions: Sequence,
) -> float:
    """Position-wise accuracy of the predicted response tokens.

    The predicted string is tokenised with the same tokeniser as the gold
    response and compared position by position over the gold response length
    (including ``<eos>``), which is the usual token-level accuracy for this task.
    """
    if len(predicted) != len(transitions):
        raise ValueError("predictions and transitions must line up")
    scores: list[float] = []
    for text, transition in zip(predicted, transitions):
        gold = tokenizer.encode_response(transition.answer)
        guess = tokenizer.encode(text) + [tokenizer.eos_id]
        hits = sum(
            1 for i, token in enumerate(gold) if i < len(guess) and guess[i] == token
        )
        scores.append(hits / len(gold))
    return float(np.mean(scores))


def evaluate_policy(
    model: WorldModel,
    tokenizer: Tokenizer,
    transitions: Sequence,
    max_new_tokens: int = 20,
    schemes: Sequence[str] = ("binary", "token_f1", "structural"),
    n_samples: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out diagnostics for one world model.

    Reports the decoded metrics under greedy decoding (the model's one-shot
    answer), the binary reward averaged over several samples, token accuracy,
    and the teacher-forced NLL of the ground-truth next states -- the
    likelihood number that RLVR is *not* optimising.
    """
    from .mle import evaluate_mle
    from .reward import score_texts

    prompts = mx.array(
        np.stack([tokenizer.encode_prompt(t.question) for t in transitions]).astype(np.int32)
    )
    prompt_len = prompts.shape[1]
    greedy = decode_rollouts(
        tokenizer,
        generate(model, prompts, max_new_tokens, tokenizer, greedy=True),
        prompt_len,
    )
    references = [t.answer for t in transitions]
    out: dict[str, float] = {}
    for scheme in schemes:
        out[f"greedy_{scheme}"] = float(np.mean(score_texts(greedy, references, scheme)))
    out["greedy_token_accuracy"] = response_token_accuracy(tokenizer, greedy, transitions)

    key = mx.random.key(seed)
    sampled_scores: list[float] = []
    for i in range(n_samples):
        key, subkey = mx.random.split(key)
        sequences = generate(
            model, prompts, max_new_tokens, tokenizer, key=subkey, temperature=1.0
        )
        texts = decode_rollouts(tokenizer, sequences, prompt_len)
        sampled_scores.append(float(np.mean(score_texts(texts, references, "binary"))))
    out["sampled_exact_match"] = float(np.mean(sampled_scores))
    out.update(evaluate_mle(model, tokenizer, transitions))
    return out
