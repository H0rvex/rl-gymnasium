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
| 1f | Run 3-seed REINFORCE + DQN training, commit metrics CSVs | [x] | _pending commit_ |
| 1g | REINFORCE README (hyperparameter table, results, plots, what I learned) | [x] | _pending commit_ |
| 1h | DQN README (same, incl. Double-DQN rationale + plots) | [x] | _pending commit_ |
| 2 | `pyproject.toml` with pinned floors; drop `requirements*.txt` | [x] | `01c6d56` |
| 3 | Root README rework (drop roadmap, update run commands to `pip install -e .`) | [x] | _pending commit_ |

### P1 — measurable improvements

| # | Item | Status | Commit |
|---|---|---|---|
| 4 | Smoke tests per algorithm (`pytest` asserting CSV written, no NaN loss) | [ ] | — |
| 5 | PPO README honest caveat (n=3 seeds, CPU-only) | [x] | _pending commit_ |
| 6 | Autoreset-fix debugging writeup in PPO README (reference commit `5e15932`) | [x] | _pending commit_ |

### P2 — nice to have

| # | Item | Status | Commit |
|---|---|---|---|
| 7 | TensorBoard or W&B logging alongside CSV | [ ] | — |
| 8 | Further consolidation in `common/` (only if duplication reappears) | [ ] | — |

## Current state

All P0 items and P1 items 5/6 are done, pending a single commit sweep. P1 item 4 (pytest smoke tests) and all P2 items are the only remaining open work.

Pending commits (in order):
1. `chore` — `.gitignore` add `logs/`, `rl_gymnasium.egg-info/`
2. `data` — 3-seed REINFORCE + DQN CSVs (`reinforce/metrics_seed*.csv`, `dqn/metrics_seed*.csv`)
3. `docs` — REINFORCE + DQN READMEs with plots, hyperparams, results, learnings (`reinforce/plots/`, `dqn/plots/`)
4. `docs(root)` — root README rework + `pyproject.toml` `requires-python` floor fix + PPO README setup fix
5. `docs(ppo)` — hardware caveat (Python 3.10) + autoreset debugging writeup
6. `docs` — this file (REFACTOR_PROGRESS.md)
