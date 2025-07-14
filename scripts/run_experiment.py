#!/usr/bin/env python
"""MLE versus MLE-then-RLVR on a toy tokenized world model.

Both stages of RLVR-World (arXiv:2505.13934) are run and compared on one small
deterministic text world:

1. MLE: the world model is trained by next-token prediction on tokenized
   ``(state, action) -> next state`` transitions (the paper's ``J_MLE``).
2. RLVR: starting from that same checkpoint, the model is post-trained with
   GRPO on a verifiable reward computed on the *decoded* next-state prediction,
   with a KL penalty to the frozen MLE model.

Every configuration is run over several seeds and evaluated on held-out
transitions, because MLX training on this machine is not bit-reproducible run
to run (identical seeds give slightly different weights).  Results are written
to ``results/run_log.txt`` and ``results/results.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from rlvr_world.grpo import RLVRConfig, train_rlvr
from rlvr_world.mle import train_mle
from rlvr_world.model import ModelConfig, WorldModel
from rlvr_world.rollout import evaluate_policy
from rlvr_world.tokenizer import Tokenizer
from rlvr_world.world import (
    GridWorld,
    destination_coverage,
    split_transitions,
    unchanged_fraction,
)

PAPER_TITLE = "RLVR-World: Training World Models with Reinforcement Learning"
PAPER_ID = "arXiv:2505.13934"
METRIC_KEYS = (
    "greedy_binary",
    "greedy_token_f1",
    "greedy_structural",
    "greedy_token_accuracy",
    "sampled_exact_match",
    "teacher_forced_nll",
    "teacher_forced_token_accuracy",
)


class Logger:
    """Prints immediately and keeps every line for ``results/run_log.txt``."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mle-steps", type=int, default=250, help="main MLE budget")
    parser.add_argument("--mle-lr", type=float, default=1e-3)
    parser.add_argument("--mle-batch-size", type=int, default=32)
    parser.add_argument(
        "--sensitivity-steps",
        type=int,
        nargs="*",
        default=[250, 500, 1000, 2000],
        help="MLE budgets for the training-length sensitivity table",
    )
    parser.add_argument("--rlvr-steps", type=int, default=300)
    parser.add_argument("--rlvr-batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=1e-3, help="the paper's 1e-3")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=["binary:1e-4", "binary:3e-5", "binary:3e-4", "token_f1:1e-4", "structural:1e-4"],
        help="RLVR configurations, each 'reward_scheme:learning_rate'",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def evaluate(
    model: WorldModel,
    tokenizer: Tokenizer,
    transitions,
    max_new_tokens: int,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    return evaluate_policy(
        model,
        tokenizer,
        transitions,
        max_new_tokens=max_new_tokens,
        n_samples=n_samples,
        seed=seed,
    )


def summarise(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [record[key] for record in per_seed]
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out[key] = {"mean": float(np.mean(values)), "std": spread}
    return out


def parse_runs(runs: list[str]) -> list[tuple[str, str, float]]:
    """Turn ``['binary:1e-4']`` into ``[('binary@1e-4', 'binary', 1e-4)]``."""
    parsed: list[tuple[str, str, float]] = []
    for spec in runs:
        scheme, _, lr = spec.partition(":")
        scheme = scheme.strip()
        if scheme not in ("binary", "token_f1", "structural"):
            raise SystemExit(f"unknown reward scheme {scheme!r} in --runs {spec!r}")
        parsed.append((f"{scheme}@{lr or '1e-4'}", scheme, float(lr or 1e-4)))
    return parsed


def format_row(name: str, summary: dict[str, dict[str, float]]) -> str:
    return (
        f"{name:<26}"
        f"{summary['greedy_binary']['mean']:.3f} +/- {summary['greedy_binary']['std']:.3f}   "
        f"{summary['greedy_token_f1']['mean']:.3f}   "
        f"{summary['greedy_token_accuracy']['mean']:.3f}   "
        f"{summary['sampled_exact_match']['mean']:.3f}   "
        f"{summary['teacher_forced_nll']['mean']:.3f}"
    )


def main() -> None:
    args = parse_args()
    log = Logger()
    started = time.time()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log(f"{PAPER_TITLE} ({PAPER_ID}) -- toy reimplementation experiment")
    log(f"device={mx.default_device()}  python={platform.python_version()}  mlx={mx.__version__}")
    log(
        "note: MLX training is not bit-reproducible run to run on this machine, so every "
        "configuration is repeated over seeds and reported as mean +/- std"
    )
    log("")

    tokenizer = Tokenizer()
    world = GridWorld(rows=args.rows, cols=args.cols, seed=args.world_seed)
    train, held_out = split_transitions(world.transitions(), args.eval_frac, seed=0)
    max_new_tokens = len(tokenizer.encode_response(train[0].answer))
    model_cfg = ModelConfig(
        vocab_size=len(tokenizer), d_model=96, n_layers=3, n_heads=4, d_ff=256, max_len=64
    )
    mx.random.seed(0)
    num_params = WorldModel(model_cfg).num_params()

    world_info = {
        "rows": args.rows,
        "cols": args.cols,
        "wall_frac": 0.12,
        "world_seed": args.world_seed,
        "train_transitions": len(train),
        "held_out_transitions": len(held_out),
        "unchanged_fraction": unchanged_fraction(world.transitions()),
        "held_out_destination_covered_by_train": destination_coverage(train, held_out),
        "vocab_size": len(tokenizer),
        "model_params": num_params,
        "response_tokens": max_new_tokens,
    }
    log(f"world: {args.rows}x{args.cols} grid, {len(train)} train / {len(held_out)} held-out transitions")
    log(
        f"  unchanged transitions {world_info['unchanged_fraction']:.3f}; "
        f"held-out destinations seen in train {world_info['held_out_destination_covered_by_train']:.3f}"
    )
    log(f"model: {num_params} parameters, vocab {len(tokenizer)}, response length {max_new_tokens}")
    log("")

    # ---------------------------------------------------------------- MLE length
    log("MLE training-length sensitivity (held-out, greedy decoding)")
    log(f"  {'steps':>6}  {'exact_match':>12}  {'token_f1':>9}  {'tok_acc':>8}  {'nll':>7}")
    sensitivity: list[dict[str, Any]] = []
    for steps in args.sensitivity_steps:
        per_seed = []
        for seed in range(args.seeds):
            mx.random.seed(seed)
            model = WorldModel(model_cfg)
            if steps > 0:
                train_mle(
                    model,
                    tokenizer,
                    train,
                    steps=steps,
                    batch_size=args.mle_batch_size,
                    lr=args.mle_lr,
                    seed=seed,
                    log_every=max(steps, 1),
                    log=lambda _: None,
                )
            per_seed.append(
                evaluate(model, tokenizer, held_out, max_new_tokens, args.eval_samples, seed)
            )
        summary = summarise(per_seed)
        sensitivity.append({"mle_steps": steps, "summary": summary})
        log(
            f"  {steps:>6}  {summary['greedy_binary']['mean']:>7.3f} +/- {summary['greedy_binary']['std']:.3f}"
            f"  {summary['greedy_token_f1']['mean']:>9.3f}  "
            f"{summary['greedy_token_accuracy']['mean']:>8.3f}  "
            f"{summary['teacher_forced_nll']['mean']:>7.3f}"
        )
    log("")

    # ------------------------------------------------------------------ main run
    log(f"main comparison: MLE {args.mle_steps} steps, then {args.rlvr_steps} RLVR steps")
    per_seed_records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        log(f"-- seed {seed}")
        mx.random.seed(seed)
        mle_model = WorldModel(model_cfg)
        train_mle(
            mle_model,
            tokenizer,
            train,
            steps=args.mle_steps,
            batch_size=args.mle_batch_size,
            lr=args.mle_lr,
            seed=seed,
            log_every=max(args.mle_steps // 2, 1),
            log=log,
        )
        mle_params = tree_map(lambda a: mx.array(a), mle_model.parameters())
        mle_metrics = evaluate(
            mle_model, tokenizer, held_out, max_new_tokens, args.eval_samples, seed
        )
        log(
            f"   MLE only            exact-match {mle_metrics['greedy_binary']:.3f}  "
            f"token-F1 {mle_metrics['greedy_token_f1']:.3f}  "
            f"token-acc {mle_metrics['greedy_token_accuracy']:.3f}  "
            f"nll {mle_metrics['teacher_forced_nll']:.3f}"
        )
        record: dict[str, Any] = {"seed": seed, "mle": mle_metrics, "rlvr": {}}

        for name, scheme, lr in parse_runs(args.runs):
            model = WorldModel(model_cfg)
            model.update(tree_map(lambda a: mx.array(a), mle_params))
            reference = WorldModel(model_cfg)
            reference.update(tree_map(lambda a: mx.array(a), mle_params))
            cfg = RLVRConfig(
                rl_coef=1.0,
                kl_coef=args.kl_coef,
                mle_coef=0.0,
                max_new_tokens=max_new_tokens,
            )
            curve: list[dict[str, Any]] = [{"step": 0, **mle_metrics}]

            def record_step(step: int, current: WorldModel, _curve=curve) -> None:
                if step % args.eval_every == 0:
                    _curve.append(
                        {
                            "step": step,
                            **evaluate(
                                current,
                                tokenizer,
                                held_out,
                                max_new_tokens,
                                args.eval_samples,
                                seed,
                            ),
                        }
                    )

            history = train_rlvr(
                model,
                reference,
                tokenizer,
                train,
                steps=args.rlvr_steps,
                batch_size=args.rlvr_batch_size,
                group_size=args.group_size,
                lr=lr,
                weight_decay=0.0,
                seed=seed,
                cfg=cfg,
                scheme=scheme,
                log_every=args.eval_every,
                log=log,
                on_step=record_step,
            )
            metrics = evaluate(
                model, tokenizer, held_out, max_new_tokens, args.eval_samples, seed
            )
            record["rlvr"][name] = {
                "metrics": metrics,
                "mean_training_reward": float(np.mean(history.reward)),
                "final_training_reward": float(history.reward[-1]),
                "mean_group_signal_fraction": float(np.mean(history.signal)),
                "final_kl": float(history.kl[-1]),
                "curve": curve,
            }
            log(
                f"   MLE + RLVR ({name:<16}) exact-match {metrics['greedy_binary']:.3f}  "
                f"token-F1 {metrics['greedy_token_f1']:.3f}  "
                f"token-acc {metrics['greedy_token_accuracy']:.3f}  "
                f"nll {metrics['teacher_forced_nll']:.3f}   "
                f"(train reward {history.reward[-1]:.3f}, groups with signal "
                f"{np.mean(history.signal):.2f}, kl {history.kl[-1]:.2e})"
            )

        # Control: the same loop with the RL coefficient at zero must not move.
        control = WorldModel(model_cfg)
        control.update(tree_map(lambda a: mx.array(a), mle_params))
        reference = WorldModel(model_cfg)
        reference.update(tree_map(lambda a: mx.array(a), mle_params))
        before = [np.array(v) for _, v in tree_flatten(control.parameters())]
        train_rlvr(
            control,
            reference,
            tokenizer,
            train,
            steps=args.rlvr_steps,
            batch_size=args.rlvr_batch_size,
            group_size=args.group_size,
            lr=1e-4,
            weight_decay=0.0,
            seed=seed,
            cfg=RLVRConfig(rl_coef=0.0, kl_coef=0.0, mle_coef=0.0, max_new_tokens=max_new_tokens),
            scheme="binary",
            log_every=args.eval_every,
            log=lambda _: None,
        )
        after = [np.array(v) for _, v in tree_flatten(control.parameters())]
        unchanged = all(np.array_equal(a, b) for a, b in zip(before, after))
        control_metrics = evaluate(
            control, tokenizer, held_out, max_new_tokens, args.eval_samples, seed
        )
        record["control_rl_coef_zero"] = {
            "weights_unchanged": bool(unchanged),
            "metrics": control_metrics,
        }
        log(
            f"   control (rl_coef=0)  weights unchanged: {unchanged}; "
            f"exact-match {control_metrics['greedy_binary']:.3f}"
        )
        per_seed_records.append(record)
    log("")

    # ------------------------------------------------------------------ summary
    run_specs = parse_runs(args.runs)
    names = ["mle"] + [name for name, _, _ in run_specs] + ["control_rl_coef_zero"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        if name == "mle":
            seeds_metrics = [record["mle"] for record in per_seed_records]
        elif name == "control_rl_coef_zero":
            seeds_metrics = [record["control_rl_coef_zero"]["metrics"] for record in per_seed_records]
        else:
            seeds_metrics = [record["rlvr"][name]["metrics"] for record in per_seed_records]
        summary[name] = summarise(seeds_metrics)

    header = (
        f"{'configuration':<26}{'exact match':<18}{'token F1':<9}{'tok acc':<9}"
        f"{'sampled EM':<13}{'NLL':<7}"
    )
    log("held-out results, mean +/- std over "
        f"{args.seeds} seeds ({len(held_out)} held-out transitions)")
    log(header)
    log("-" * len(header))
    for name in names:
        label = {
            "mle": "MLE only",
            "control_rl_coef_zero": "control: rl_coef=0",
        }.get(name, f"MLE + RLVR ({name})")
        log(format_row(label, summary[name]))
    log("")
    log("paired per-seed change in held-out exact match (RLVR - MLE)")
    for name in [name for name, _, _ in run_specs]:
        deltas = [
            record["rlvr"][name]["metrics"]["greedy_binary"] - record["mle"]["greedy_binary"]
            for record in per_seed_records
        ]
        log(
            f"  {name:<16} " + "  ".join(f"{d:+.3f}" for d in deltas)
            + f"   mean {np.mean(deltas):+.3f}"
        )
    log("")
    log(f"total wall clock {time.time() - started:.0f} s")

    results = {
        "paper": {"title": PAPER_TITLE, "arxiv_id": PAPER_ID},
        "disclaimer": (
            "Toy-scale independent implementation. Numbers are not comparable to the "
            "paper's, which uses DeepSeek-R1-Distill-Qwen-1.5B on ByteSized32 and WebArena."
        ),
        "env": {
            "device": str(mx.default_device()),
            "mlx": mx.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "config": vars(args) | {"results_dir": str(args.results_dir)},
        "world": world_info,
        "mle_length_sensitivity": sensitivity,
        "per_seed": per_seed_records,
        "summary": summary,
        "wall_clock_seconds": time.time() - started,
    }
    (args.results_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    log.save(args.results_dir / "run_log.txt")
    log(f"wrote {args.results_dir / 'results.json'} and {args.results_dir / 'run_log.txt'}")


if __name__ == "__main__":
    main()
