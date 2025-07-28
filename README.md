# rlvr-world-mlx

A small MLX reimplementation of *RLVR-World: Training World Models with Reinforcement Learning* (arXiv:2505.13934). The paper's observation is that a world model is trained by maximum likelihood on tokenized transitions, but the thing we actually want from it is a metric on the **decoded** prediction — exact-match accuracy for text states, a perceptual or structural metric for video. That metric sits on the far side of a decode step and is not differentiable, so the paper post-trains the model with reinforcement learning with verifiable rewards (RLVR): sample rollouts from the model, decode them, score the decoded predictions with a programmatic reward, and take GRPO steps on that reward with a KL penalty to the frozen maximum-likelihood model.

This repository is an independent, toy-scale implementation written from the paper's description. It is not a reproduction of the paper's experiments or numbers. The paper post-trains DeepSeek-R1-Distill-Qwen-1.5B on ByteSized32 game-state transitions and web-page accessibility trees, and an autoregressive video model on robot-manipulation trajectories, reporting gains such as +30.7% accuracy and +15.1% F1. The world model here has 277,344 parameters, the world is a deterministic 8×8 grid rendered as short token strings, and both training stages run in about nine minutes on one Apple-silicon laptop. The numbers below are from that toy and are not comparable to the paper's.

## What is in here

**The task** (`rlvr_world/world.py`). A `GridWorld` is a grid with wall cells, each free cell carrying three binary properties (`door` ∈ {open, shut}, `lamp` ∈ {on, off}, `gem` ∈ {yes, no}). A state is the agent's cell plus the properties of that cell, rendered as one string, `r3 c5 door open lamp off gem yes`; an action is one of the four compass moves. The transition is deterministic: the agent moves if the destination is inside the grid and not a wall, and otherwise stays put, in which case the next state is the current state (the paper's "unchanged cases"; here 24.5% of transitions). Because the properties are a function of the cell, the rendered state carries everything needed to predict the next one, so the process is first-order Markov exactly as in the paper's sequence formulation. Four actions from every free cell give the transition set, which is split 176 train / 44 held-out with a fixed seed. Every held-out transition has a destination cell that does occur in training, so the held-out task is generalisation over `(state, action)` pairs, not memorisation of unseen cells.

**The tokeniser** (`rlvr_world/tokenizer.py`). A greedy longest-match tokeniser over a 56-piece vocabulary: special pieces (`<pad> <bos> <sep> <eos> <unk>`), single characters, and multi-character keyword pieces (`door`, `lamp`, `gem`, `open`, `shut`, `north`, …). The same string therefore has several tokenisations — `encode("door open")` gives three pieces, `encode_charwise` gives ten — and both decode to the same text. That is deliberate: it makes the paper's central claim, that the reward is a metric on the decoded prediction rather than on the token ids, directly testable. A question is `<bos> q(s, a) <sep>` and a response is `o(s') <eos>`, 18 tokens for the response at the default configuration.

**The reward** (`rlvr_world/reward.py`). This is the paper's Eq. 3, `R_i = sign(D) · D(decode(o_i), s')`, with three metrics in `[0, 1]`:

- `binary` — exact match of the extracted state, the paper's binary reward for text game state prediction;
- `token_f1` — token-level F1 between the canonical renderings, in the spirit of the paper's F1 reward for web page state prediction;
- `structural` — fraction of the five state fields that match, a dense structural distance `1 - d`.

Nothing looks at token ids: every reward first decodes to text and then extracts the state with `parse_state`, a rule-based extractor that scans for each field independently, so field order, whitespace and `=` separators do not matter — the same tolerance the paper's rule-based extractor implies. A prediction whose state cannot be extracted scores 0 under all three metrics.

**The world model** (`rlvr_world/model.py`). A decoder-only transformer: learned token and position embeddings, three pre-norm blocks, four heads over width 96, a 256-wide GELU MLP, and an un-tied output head — 277,344 parameters. Attention is written out by hand (no fused kernel) so the tests can probe the causal mask directly.

**MLE stage** (`rlvr_world/mle.py`). The paper's Eq. 2, `J_MLE(θ) = Σ_t log p_θ(o_t(s') | q(s,a), o_<t(s'))`, computed as a mean negative log-likelihood with the prompt positions masked out of the loss (the model still reads them). AdamW, batch 32, learning rate 1e-3, weight decay 0.01, 250 steps for the main comparison.

**RLVR stage** (`rlvr_world/rollout.py`, `rlvr_world/grpo.py`). For each training question the model samples a group of `G = 8` responses at temperature 1.0 with structural pieces banned (so a rollout is always a decodable string, and `<eos>` stops it). Each rollout is decoded and scored, and the group's rewards are normalised into advantages, `Â_i = (R_i - mean(R)) / std(R)` (Eq. 1), with the degenerate group — all rewards equal — mapped to zero advantage rather than dividing by ~0. The objective is the clipped GRPO surrogate with a KL penalty,

```
J(θ) = E[ mean_i mean_t ( min(ρ_i,t Â_i, clip(ρ_i,t, 1-ε, 1+ε) Â_i) - β D_KL[p_θ ‖ p_ref] ) ]
```

with `ρ = p_θ / p_θ_old`, `ε = 0.2`, `β = 1e-3` (the paper's KL coefficient) and `<eos>` included in the scored tokens. The KL is the per-token `k3` estimator used by GRPO implementations, `exp(log p_ref - log p_θ) - (log p_ref - log p_θ) - 1`, whose expectation under the policy is exactly `D_KL[p_θ ‖ p_ref]`; the tests check that against the full-vocabulary KL by enumerating a small vocabulary. The reference model is a frozen copy of the MLE checkpoint. Two coefficients are ours and are not in the paper: `rl_coef` multiplies the whole RL term and `mle_coef` adds an optional supervised anchor (`0` by default, so the paper's RLVR stage is `rl_coef = 1, mle_coef = 0`). They exist so that the limits in the tests can be stated exactly.

## Checks

`tests/` has 50 tests. The ones that would catch a real bug:

- **The reward is a function of the decoded string, not the tokens.** The same prediction is scored through its keyword tokenisation and through its character tokenisation, which are different id sequences with different lengths, and through reordered/spaced/`=`-separated text; all give identical rewards under all three metrics (`test_reward.py`).
- **Hand-computed rewards**: 1.0 for a perfect prediction, 0.0 for each of five single-field errors under `binary`, `7/8` under token F1 for one wrong field, `4/5` under `structural`, and 0.0 for an unparseable prediction.
- **Hand-computed advantages**: `[1, 2, 3, 4]` gives `[-1.5, -0.5, 0.5, 1.5] / sqrt(5/3)`; a constant group gives zeros; shifting and scaling the rewards leaves the advantages unchanged.
- **KL penalty**: the `k3` estimator is elementwise zero when policy and reference are the same distribution, and its expectation equals the exact `D_KL[q ‖ p]` computed by enumeration over a four-symbol vocabulary.
- **The RLVR limit**: with `mle_coef = 1, rl_coef = 0, kl_coef = 0` the RLVR loss equals `mle_loss` to floating point, with identical gradients; with `rl_coef = 0` and no anchor the loss is exactly 0.0 and every gradient is exactly 0.0, and `train_rlvr` with that configuration leaves the parameters bit-identical (the control in the experiment reproduces this end-to-end).
- **The clipped surrogate**, hand-computed for positive and negative advantages and ratios inside and outside the clip; the clip fraction is checked too.
- **Gradients are finite** for both the MLE and the RLVR objective, including an off-policy ratio.
- **Causality**: changing a later token does not change earlier logits; `mle_loss` equals a `mx.fast.cross_entropy` reference computed independently; a uniform model gives `log(vocab_size)`.
- **World invariants**: transitions are deterministic, blocked moves are identities, the rendered properties always match the cell tables, and the train/held-out split is disjoint, complete and reproducible.

## Experiment

`scripts/run_experiment.py` builds the world, trains the model by MLE, then post-trains a copy of that checkpoint with RLVR and compares both on the 44 held-out transitions. Held-out metrics are decoded metrics under greedy decoding (the model's one-shot answer), the binary reward averaged over 8 samples, the token-level accuracy of the predicted response, and the teacher-forced NLL of the ground-truth next states — the likelihood that RLVR is not optimising.

Because MLX training on this machine is not bit-reproducible run to run — the 250-step sensitivity row and the MLE row below are the same configuration, same seeds and same budget, run twice, and differ by 0.022 — every configuration is repeated over three seeds and reported as mean ± std, and the per-seed paired change is reported as well. The whole run takes about nine minutes (`results/run_log.txt` ends with the wall clock, 530 s).

MLE budget: 250 steps at lr 1e-3. Held-out results, mean ± std over three seeds (44 held-out transitions, so one transition is 0.023):

| configuration | exact match | token F1 | token acc | sampled exact match | NLL (nats) |
|---|---:|---:|---:|---:|---:|
| MLE only | 0.326 ± 0.112 | 0.870 | 0.942 | 0.227 | 0.123 |
| MLE + RLVR, binary reward, lr 1e-4 | **0.477 ± 0.202** | 0.895 | 0.953 | 0.429 | 0.214 |
| MLE + RLVR, binary reward, lr 3e-5 | 0.455 ± 0.157 | 0.891 | 0.953 | 0.405 | 0.156 |
| MLE + RLVR, binary reward, lr 3e-4 | 0.121 ± 0.057 | 0.705 | 0.865 | 0.123 | 0.754 |
| MLE + RLVR, token F1 reward, lr 1e-4 | 0.394 ± 0.086 | 0.886 | 0.949 | 0.343 | 0.233 |
| MLE + RLVR, structural reward, lr 1e-4 | 0.462 ± 0.139 | 0.899 | 0.955 | 0.426 | 0.208 |
| control: RLVR loop with `rl_coef = 0` | 0.326 ± 0.112 | 0.870 | 0.942 | 0.227 | 0.123 |

Paired per-seed change in held-out exact match (RLVR − MLE): binary at 1e-4 `+0.250, +0.068, +0.136`; binary at 3e-5 `+0.182, +0.114, +0.091`; binary at 3e-4 `−0.273, −0.136, −0.205`; token F1 `+0.000, +0.045, +0.159`; structural `+0.159, +0.091, +0.159`. Every non-diverging configuration improved on at least two of the three seeds. The RLVR control with `rl_coef = 0` left the weights bit-identical for all three seeds and reproduces the MLE row exactly.

What the numbers say:

- **RLVR does raise the decoded metric.** With the binary reward and lr 1e-4, held-out exact match rises from 0.326 to 0.477 (paired mean +0.152, all three seeds up) and token-level accuracy from 0.942 to 0.953. The sampled exact match, which is what the RLVR reward actually sees during training, rises from 0.227 to 0.429, and the sampled training reward on the training questions from 0.406 / 0.297 / 0.078 at the first step to 0.953 / 0.891 / 0.688 at the end (seeds 0/1/2, as printed in `results/run_log.txt`).
- **It costs likelihood, as expected.** Teacher-forced NLL rises from 0.123 to 0.214 nats. The same effect is visible at a smaller step size, where the metric gain is a little smaller and the NLL cost much smaller (0.156 at lr 3e-5), and it becomes catastrophic at lr 3e-4, where the KL penalty reaches 0.210 / 0.403 / 0.793 by the end (seeds 0/1/2) and both the metric and the likelihood collapse.
- **Denser rewards are not automatically better here.** The token-F1 reward improves exact match less (+0.068) while costing more likelihood (0.233), which is what one would expect from a reward that pays for partially correct answers. The structural reward behaves like the binary one (+0.136).
- **More MLE is a stronger baseline than RLVR in this toy.** The training-length sensitivity in the same run, on the same seeds and held-out set, gives held-out exact match 0.348 ± 0.086 at 250 MLE steps, 0.621 ± 0.148 at 500, 0.705 ± 0.114 at 1000 and 0.735 ± 0.129 at 2000. RLVR's 300 steps took the 250-step checkpoint to 0.477, which is better than where it started but well short of simply training the likelihood five times longer. (The 250-step sensitivity row, 0.348, differs from the MLE row, 0.326, only because of the run-to-run non-reproducibility above — the two are separate runs of the same configuration.)
- The GRPO loss printed in the log is ≈ 0 by construction, because group-normalised advantages sum to zero within each group; the informative training diagnostics are the sampled reward and the KL.
- The held-out set has 44 transitions, so one transition is 0.023 and differences of one or two transitions are noise. The seed-to-seed spread is comparable to the effect being measured, which is why the paired per-seed numbers are quoted next to the means.

The honest summary is that RLVR improved the decoded metric from a fixed checkpoint on every seed in this toy, at a real likelihood cost, but it did not beat simply continuing likelihood training — a comparison the paper's much larger setting does not face in the same way.

## Where this may differ from the paper

- **Scale and domain.** 277k parameters on a synthetic 8×8 grid, against DeepSeek-R1-Distill-Qwen-1.5B with LoRA on ByteSized32 and WebArena, and an autoregressive video world model with a learned visual tokenizer. There is no video, no visual codebook, no perceptual metric and no multi-step horizon here; the Markov order is 1.
- **The task formulation.** The paper trains the text-world model on state *differences* and uses a task-specific reward `0.1·acc_all + 1·acc_changed + 0.2·𝕀(correct)`. Here the model predicts the absolute next state and the reward is a metric on that whole string. The discount structure of the paper's reward is not implemented; `binary`, `token_f1` and `structural` are stand-ins for the binary, F1 and perceptual metrics the paper uses across its domains.
- **The reward extractor is mine.** `parse_state` is order-, whitespace- and separator-insensitive. The paper's rule-based extractor is described at the level of "the model must produce the exact same string including all characters and formatting" for web-page item changes, which is stricter than what I do for the text-game states; a prediction that lists the same fields in a different order scores 1.0 here.
- **KL penalty.** I use the per-token `k3` estimator (the GRPO convention) rather than the exact per-step KL, and I normalise by each response's own length as in Eq. 1. When policy and reference coincide the estimator is exactly zero, which is what the limit test exploits.
- **One inner epoch per batch.** Each batch is sampled once and used for a single gradient step, so the importance ratio is exactly 1 at the loss and the PPO clip never binds during the experiment; the clip is implemented and unit-tested on synthetic ratios, but the run does not exercise it. There is no minibatching within a batch either. The paper's RLVR uses batch 128 with GRPO minibatches of 64.
- **Hyperparameters.** Group size 8 (the paper uses 5 for text games and 16 for video), temperature 1.0 with no top-p truncation (which matches the paper), KL coefficient 1e-3 (the paper's), AdamW at lr 1e-4 with weight decay 0 for RLVR (the paper uses 1e-6 and 0.01, on a LoRA-adapted 1.5B model), 300 RLVR steps (the video results need "a few hundred"; the paper's language runs are longer). The MLE stage uses 250 steps at lr 1e-3, chosen so that the baseline is a competent but unsaturated fit — see the sensitivity table for what longer MLE does.
- **The `rl_coef` and `mle_coef` coefficients are additions**, not the paper's formulation. They exist so that "RLVR with the RL coefficient at zero reduces to MLE / is a no-op" can be tested exactly, and they are set to `1` and `0` in every reported RLVR result.
- **No multiple-inner-epoch stability tricks, no reward shaping, no advantage whitening beyond the group normalisation, no curriculum.** The sample counts are small enough that the seed noise is large; I report it rather than hiding it behind a single seed.
- **Non-reproducibility.** MLX training here is deterministic in initialisation but not in the training loop: identical seeds give slightly different weights, so exact numbers will shift if the experiment is re-run. Everything is seeded (`mx.random.seed`, `np.random.default_rng`, split seeds) and the variation is reported as std over seeds.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest -q
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/run_experiment.py \
  --seeds 3 --mle-steps 250 --mle-lr 1e-3 \
  --sensitivity-steps 250 500 1000 2000 \
  --rlvr-steps 300 --eval-every 100 --eval-samples 8 \
  --runs binary:1e-4 binary:3e-5 binary:3e-4 token_f1:1e-4 structural:1e-4
```

Both commands are run from the repository root; the experiment script adds the repository root to `sys.path` itself, so no install step is needed. `--runs` takes `reward_scheme:learning_rate` pairs and `--results-dir` defaults to `results/`, where the script overwrites `run_log.txt` and `results.json` with the run it just did. A single-seed run with one configuration (`--seeds 1 --runs binary:1e-4`) finishes in about a minute, which is the quickest way to see the pipeline work.
