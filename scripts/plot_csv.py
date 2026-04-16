import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    # Minimal CSV reader (no pandas dependency)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("CSV is empty")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to a metrics CSV file")
    p.add_argument(
        "--x",
        type=str,
        default="iteration",
        help="X-axis column (default: iteration)",
    )
    p.add_argument(
        "--ys",
        type=str,
        default="eval_det_mean,eval_sto_mean",
        help="Comma-separated Y columns to plot",
    )
    p.add_argument("--out", type=str, default="", help="Output image path (optional)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    cols = read_csv(csv_path)
    if args.x not in cols:
        raise SystemExit(f"Missing x column '{args.x}'. Columns: {list(cols.keys())}")

    x = to_float(cols[args.x])
    ys = [y.strip() for y in args.ys.split(",") if y.strip()]

    plt.figure(figsize=(9, 5))
    for yk in ys:
        if yk not in cols:
            raise SystemExit(f"Missing y column '{yk}'. Columns: {list(cols.keys())}")
        y = to_float(cols[yk])
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            raise SystemExit(f"No finite points to plot for '{yk}'.")
        # Use markers so sparse series (e.g. eval every N iters) still show up.
        plt.plot(x[mask], y[mask], marker="o", markersize=3, linewidth=1.2, label=yk)

    plt.title(csv_path.name)
    plt.xlabel(args.x)
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

