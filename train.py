import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parent


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training entrypoint.")
    parser.add_argument("--algorithm", choices=["reinforce", "dqn", "ppo"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Run artifact directory. Defaults to outputs/runs/{algorithm}_seed{seed}.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="REINFORCE/DQN episode budget.")
    parser.add_argument("--iterations", type=int, default=None, help="PPO iteration budget.")
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DQN only: use Double DQN targets by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or str(ROOT / "outputs" / "runs" / f"{args.algorithm}_seed{args.seed}")

    if args.algorithm == "reinforce":
        module = _load_module("reinforce_train", ROOT / "reinforce" / "train.py")
        argv = ["--seed", str(args.seed), "--device", args.device, "--out-dir", out_dir]
        if args.episodes is not None:
            argv.extend(["--episodes", str(args.episodes)])
    elif args.algorithm == "dqn":
        module = _load_module("dqn_train", ROOT / "dqn" / "train.py")
        argv = ["--seed", str(args.seed), "--device", args.device, "--out-dir", out_dir]
        if args.episodes is not None:
            argv.extend(["--episodes", str(args.episodes)])
        argv.append("--double-dqn" if args.double_dqn else "--no-double-dqn")
    else:
        module = _load_module("ppo_train", ROOT / "ppo" / "train.py")
        argv = ["--seed", str(args.seed), "--device", args.device, "--out-dir", out_dir]
        if args.iterations is not None:
            argv.extend(["--iterations", str(args.iterations)])

    module.main(module.parse_args(argv))


if __name__ == "__main__":
    main()
