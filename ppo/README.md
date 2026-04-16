## PPO (LunarLander-v3)

From-scratch PPO implementation in PyTorch + Gymnasium, using vectorized rollouts and standard PPO stability tricks:
- masked GAE
- value function clipping
- observation normalization (running mean/std)
- consistent evaluation (frozen normalization snapshot)
- vectorized rollout collection (8 environments)

### Training setup (current defaults)
- Env: `LunarLander-v3`
- Parallel envs: `8`
- Rollout length: `256` steps per env (`2048` transitions/iteration)
- PPO update: `epochs=8`, `batch_size=256`, `clip=0.2`
- Optimizer: Adam (`lr=3e-4` with floor `1e-4`, `eps=1e-5`)
- Schedules:
  - entropy: higher early (`0.01 * frac`) then lower after iteration 300 (`0.005 * frac`)
  - learning rate: linear decay with floor (`max(1e-4, 3e-4 * frac)`)

### Run
From repo root:

```bash
python ppo/train.py
```

### Output
- Prints rollout episode returns, approximate KL, and periodic evaluation.
- Saves the best model checkpoint to `best_ppo_lunarlander.pt` (ignored by git).
- Writes metrics CSV to `ppo/metrics.csv`.

### Plot training curves
From repo root:

```bash
python scripts/plot_csv.py --csv ppo/metrics.csv --ys eval_det_mean,eval_sto_mean
python scripts/plot_csv.py --csv ppo/metrics.csv --ys rollout_ep_ret_mean
```

### Latest run snapshot
- Consistently solved LunarLander by end of training.
- Example late-run eval:
  - deterministic: ~`226.5 ± 53.0`
  - stochastic: ~`232.1 ± 57.2`

