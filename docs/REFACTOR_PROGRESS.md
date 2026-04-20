# Portfolio refactor progress

Living doc for the rl-gymnasium portfolio-readiness refactor. Update as work lands.

## Branch

`portfolio-refactor`

## Decisions

- **Algorithm scope:** lift REINFORCE + DQN to PPO caliber (not cut). Three algorithms, one cohesive repo.
- **DQN variant:** `--double-dqn` flag, defaults `True`. `--no-double-dqn` gives vanilla. Labelled "Double DQN" in README.
- **Roadmap:** drop the "📋 planned" rows entirely. Portfolio shows shipped work, not intentions.
- **Eval cadence:** REINFORCE every 25 eps, DQN every 50 eps.
- **DQN target update:** per-env-step, every 1000 steps (was: per 10 episodes).
- **Scaffolding:** Option C — minimal `common/` package for zero-policy utilities (device, seeding, CSV logger). Networks / update rules / eval stay in-file.
- **Packaging:** `pyproject.toml` + `pip install -e .` replaces `requirements.txt`. `[box2d]` and `[dev]` as optional extras.

## Action list

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

### P0 — portfolio blockers

| # | Item | Status | Commit |
|---|---|---|---|
| 1a | REINFORCE scaffolding (dataclass, argparse, CSV, eval, checkpoint) | [x] | `01c6d56` |
| 1b | DQN scaffolding (same + `--double-dqn` flag, per-step target update) | [x] | `01c6d56` |
| 1c | `common/` package (device, seeding, CsvLogger) | [x] | `01c6d56` |
| 1d | Refactor all three train.py to import from `common/` | [x] | `01c6d56` |
| 1e | Smoke-run the three refactored train.py files (10-iter sanity) | [x] | _no artifacts (cleaned)_ |
| 1f | Run 3-seed REINFORCE + DQN training, commit metrics CSVs | [~] | _pending commit_ |
| 1g | REINFORCE README (hyperparameter table, results, plots, what I learned) | [x] | _pending commit_ |
| 1h | DQN README (same, incl. Double-DQN rationale + plots) | [x] | _pending commit_ |
| 2 | `pyproject.toml` with pinned floors; drop `requirements*.txt` | [x] | `01c6d56` |
| 3 | Root README rework (drop roadmap, update run commands to `pip install -e .`) | [ ] | — |

### P1 — measurable improvements

| # | Item | Status | Commit |
|---|---|---|---|
| 4 | Smoke tests per algorithm (`pytest` asserting CSV written, no NaN loss) | [ ] | — |
| 5 | PPO README honest caveat (n=3 seeds, CPU-only) | [ ] | — |
| 6 | Autoreset-fix debugging writeup in PPO README (reference commit `5e15932`) | [ ] | — |

### P2 — nice to have

| # | Item | Status | Commit |
|---|---|---|---|
| 7 | TensorBoard or W&B logging alongside CSV | [ ] | — |
| 8 | Further consolidation in `common/` (only if duplication reappears) | [ ] | — |

## Current session

Scaffolding refactor landed as `01c6d56`. 1e smoke-runs passed for all three trainers (REINFORCE 10 eps, DQN 5 eps, PPO 2 iters) — smoke artifacts cleaned. 1f completed: 3 seeds × REINFORCE (1000 eps) and 3 seeds × DQN (2000 eps, Double DQN default) on CartPole-v1, CPU. Results:

- **REINFORCE** best_det / final_det per seed: 500/500, 500/493, 500/480 — solved on all seeds, mild final-epoch oscillation
- **DQN** best_det / final_det per seed: 500/500, 500/237, 500/268 — all seeds hit 500 briefly, seeds 1/2 regressed (classic DQN instability worth documenting in 1h)

CSVs staged but uncommitted pending commit-plan review.

Env note: `pyproject.toml` requires `python>=3.11` but the active `ml` conda env is Python 3.10. Trainers run fine under 3.10 (no 3.11-only syntax). Decide: lower `requires-python` floor to `3.10`, or build a 3.11 env. Runs here used `PYTHONPATH=.` since `pip install -e .` refuses under 3.10.

## Next session

Commit the 3-seed CSVs (see commit plan), then start **1g** (REINFORCE README) and **1h** (DQN README — the seed 1/2 regression is the headline finding, motivates Double DQN rationale). After that, **3** (root README rework).
