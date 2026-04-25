import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    # Minimal CSV reader (no pandas dependency)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"CSV {path} is empty")
    header = [h.strip() for h in lines[0].split(",")]
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    cols = {h: [] for h in header}
    for row in rows:
        for h, v in zip(header, row):
            cols[h].append(v.strip())
    return cols


def to_float(xs):
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def smooth(y, window):
    """NaN-aware trailing rolling mean. Window=1 returns y unchanged."""
    if window <= 1:
        return y
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = y[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) > 0:
            out[i] = seg.mean()
    return out


def align_by_x(runs_x, runs_y):
    """Align (x, y) arrays from multiple runs onto the union of x values.

    Returns (x_union [K], y_matrix [n_runs, K]) with NaN where a run is missing a point.
    """
    x_union = sorted(set().union(*[set(x.tolist()) for x in runs_x]))
    x_arr = np.asarray(x_union, dtype=np.float64)
    y_arr = np.full((len(runs_x), len(x_arr)), np.nan)
    x_to_idx = {v: i for i, v in enumerate(x_arr)}
    for r, (x, y) in enumerate(zip(runs_x, runs_y)):
        for xv, yv in zip(x, y):
            if xv in x_to_idx:
                y_arr[r, x_to_idx[xv]] = yv
    return x_arr, y_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path or glob (e.g. 'ppo/metrics_seed*.csv') to one or more CSV files. "
        "If multiple files match, plots mean +/- std across runs.",
    )
    p.add_argument("--x", type=str, default="iteration", help="X-axis column (default: iteration)")
    p.add_argument(
        "--ys",
        type=str,
        default="eval_det_mean,eval_sto_mean",
        help="Comma-separated Y columns to plot",
    )
    p.add_argument(
        "--smooth", type=int, default=1, help="Trailing rolling-mean window (1 = no smoothing)"
    )
    p.add_argument(
        "--title", type=str, default="", help="Plot title (default: derived from CSV path)"
    )
    p.add_argument("--ylabel", type=str, default="", help="Y-axis label (default: empty)")
    p.add_argument(
        "--out", type=str, default="", help="Output image path (default: show interactively)"
    )
    args = p.parse_args()

    paths = sorted(Path(s) for s in glob.glob(args.csv))
    if not paths:
        # Fall back to treating --csv as a literal path (handy on shells that don't expand globs)
        literal = Path(args.csv)
        if literal.exists():
            paths = [literal]
        else:
            raise SystemExit(f"No files matched: {args.csv}")

    runs = [read_csv(p) for p in paths]
    multi = len(runs) > 1
    ys = [y.strip() for y in args.ys.split(",") if y.strip()]

    plt.figure(figsize=(9, 5))

    for yk in ys:
        runs_x, runs_y = [], []
        for r in runs:
            if args.x not in r or yk not in r:
                continue
            x = to_float(r[args.x])
            y = to_float(r[yk])
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if args.smooth > 1:
                y = smooth(y, args.smooth)
            runs_x.append(x)
            runs_y.append(y)

        if not runs_x:
            continue

        if multi:
            x_arr, y_arr = align_by_x(runs_x, runs_y)
            mean = np.nanmean(y_arr, axis=0)
            std = np.nanstd(y_arr, axis=0)
            mask = np.isfinite(mean)
            label = f"{yk} (n={len(runs)})"
            (line,) = plt.plot(x_arr[mask], mean[mask], linewidth=1.6, label=label)
            plt.fill_between(
                x_arr[mask],
                (mean - std)[mask],
                (mean + std)[mask],
                alpha=0.2,
                color=line.get_color(),
            )
        else:
            plt.plot(runs_x[0], runs_y[0], marker="o", markersize=3, linewidth=1.2, label=yk)

    if args.title:
        plt.title(args.title)
    elif multi:
        plt.title(f"{len(paths)} runs ({paths[0].parent.name}/{paths[0].stem.split('_seed')[0]})")
    else:
        plt.title(paths[0].name)

    plt.xlabel(args.x)
    if args.ylabel:
        plt.ylabel(args.ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
