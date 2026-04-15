## PPO (LunarLander-v3)

From-scratch PPO implementation in PyTorch + Gymnasium, using vectorized rollouts and standard PPO stability tricks:
- masked GAE
- value function clipping
- observation normalization (running mean/std)
- consistent evaluation (frozen normalization snapshot)

### Run
From repo root:

```bash
python ppo/train.py
```

### Output
- Prints rollout episode returns, approximate KL, and periodic evaluation.
- Saves the best model checkpoint to `best_ppo_lunarlander.pt` (ignored by git).

