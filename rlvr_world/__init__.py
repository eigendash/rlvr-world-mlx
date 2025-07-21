"""Toy-scale MLX implementation of RLVR-World (arXiv:2505.13934).

RLVR-World post-trains a tokenized world model with reinforcement learning on a
verifiable reward that is computed on the *decoded* next-state prediction, using
GRPO with a KL penalty to the maximum-likelihood reference model.

The public API, module by module:

- :mod:`rlvr_world.tokenizer` -- :class:`Tokenizer`: greedy keyword/character
  tokenisation, with several tokenisations of the same string.
- :mod:`rlvr_world.world` -- :class:`GridWorld`, :class:`State`,
  :class:`Transition` and :func:`split_transitions`: the deterministic toy task.
- :mod:`rlvr_world.reward` -- the verifiable rewards (:func:`exact_match_reward`,
  :func:`token_f1_reward`, :func:`structural_reward`) and :func:`parse_state`,
  the rule-based extractor that turns decoded text back into a state.
- :mod:`rlvr_world.model` -- :class:`WorldModel`, the tiny decoder-only
  transformer.
- :mod:`rlvr_world.mle` -- :func:`mle_loss` and :func:`train_mle`, the paper's
  maximum-likelihood stage.
- :mod:`rlvr_world.rollout` -- rollouts, group sampling and evaluation.
- :mod:`rlvr_world.grpo` -- :func:`group_advantages`, :func:`clipped_surrogate`,
  :func:`kl_k3`, :func:`rlvr_loss` and :func:`train_rlvr`, the RLVR stage.
"""

from .grpo import (
    RLVRConfig,
    clipped_surrogate,
    group_advantages,
    kl_exact,
    kl_k3,
    rlvr_loss,
    train_rlvr,
)
from .mle import evaluate_mle, example_batch, mle_loss, train_mle
from .model import ModelConfig, WorldModel
from .reward import (
    REWARD_SCHEMES,
    exact_match_reward,
    parse_state,
    reward_from_text,
    reward_from_tokens,
    score_texts,
    structural_reward,
    token_f1_reward,
)
from .rollout import evaluate_policy, generate, sample_group
from .tokenizer import Tokenizer
from .world import GridWorld, State, Transition, split_transitions

__all__ = [
    "Tokenizer",
    "GridWorld",
    "State",
    "Transition",
    "split_transitions",
    "REWARD_SCHEMES",
    "parse_state",
    "exact_match_reward",
    "token_f1_reward",
    "structural_reward",
    "reward_from_text",
    "reward_from_tokens",
    "score_texts",
    "ModelConfig",
    "WorldModel",
    "mle_loss",
    "train_mle",
    "evaluate_mle",
    "example_batch",
    "generate",
    "sample_group",
    "evaluate_policy",
    "RLVRConfig",
    "group_advantages",
    "clipped_surrogate",
    "kl_k3",
    "kl_exact",
    "rlvr_loss",
    "train_rlvr",
]
