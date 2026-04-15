## rl-gymnasium (from-scratch RL, PyTorch + Gymnasium)

Small, from-scratch implementations of classic RL algorithms. Each folder is self-contained and runnable.

### Projects
- **REINFORCE (CartPole-v1)**: `reinforce/train.py`
- **DQN (CartPole-v1)**: `dqn/train.py`
- **PPO (LunarLander-v3)**: `ppo/train.py`

### Setup
Install the core deps:

```bash
pip install torch gymnasium
```

For LunarLander (Box2D), you may need:

```bash
pip install "gymnasium[box2d]"
```

### Run
From the repo root:

```bash
python reinforce/train.py
python dqn/train.py
python ppo/train.py
```

### Notes
- Checkpoints (e.g. `*.pt`) are ignored by git.
- For mentoring context, see `MENTOR_CONTEXT.md` (and ask the assistant to read it first in new chats).

