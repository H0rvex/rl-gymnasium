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
- **Packaging:** `pyproject.toml` + `pip install -e .` replaces `requirements.txt`. `[box2d]` and `[dev]` as optional extras. `requires-python` floor is `3.10` (code uses no 3.11-only syntax; tested on 3.10).

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
| 1f | Run 3-seed REINFORCE + DQN training, commit metrics CSVs | [x] | `9301e83` |
| 1g | REINFORCE README (hyperparameter table, results, plots, what I learned) | [x] | `9782a35` |
| 1h | DQN README (same, incl. Double-DQN rationale + plots) | [x] | `9782a35` |
| 2 | `pyproject.toml` with pinned floors; drop `requirements*.txt` | [x] | `01c6d56` |
| 3 | Root README rework (drop roadmap, update run commands to `pip install -e .`) | [x] | `4457216` |

### P1 — measurable improvements

| # | Item | Status | Commit |
|---|---|---|---|
| 4 | Smoke tests per algorithm (`pytest` asserting CSV written, no NaN loss) | [x] | `5942ac8` |
| 5 | PPO README honest caveat (n=3 seeds, CPU-only) | [x] | `aff0efa` |
| 6 | Autoreset-fix debugging writeup in PPO README (reference commit `5e15932`) | [x] | `aff0efa` |

### P2 — nice to have

| # | Item | Status | Commit |
|---|---|---|---|
| 7 | TensorBoard or W&B logging alongside CSV | [x] | `fd1b0cd` |
| 8 | Further consolidation in `common/` (only if duplication reappears) | [x] | `fd1b0cd` |

## Current state

All items done. Every commit landed on `portfolio-refactor`; branch merged to `main`.
