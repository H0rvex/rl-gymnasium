import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def prepare_run_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain_value(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _plain_value(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    return value


def _yaml_lines(data: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        value = _plain_value(value)
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_yaml_lines(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.extend(_yaml_lines(item, indent + 4))
                else:
                    lines.append(f"{prefix}  - {item}")
        elif isinstance(value, str):
            escaped = value.replace('"', '\\"')
            lines.append(f'{prefix}{key}: "{escaped}"')
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


def write_config_yaml(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(_yaml_lines(_plain_value(data))) + "\n"
    path.write_text(text, encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not np.isfinite(out):
        return None
    return out


def write_reward_curve(
    metrics_path: Path,
    out_path: Path,
    *,
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str = "Episode return",
    smooth: int = 1,
) -> None:
    rows = _read_csv(metrics_path)
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = _finite_float(row.get(x_key))
        y = _finite_float(row.get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    if xs:
        y_arr = np.asarray(ys, dtype=np.float64)
        if smooth > 1:
            smoothed = np.full(len(y_arr), np.nan)
            for i in range(len(y_arr)):
                lo = max(0, i - smooth + 1)
                smoothed[i] = np.nanmean(y_arr[lo : i + 1])
            y_arr = smoothed
        plt.plot(xs, y_arr, linewidth=1.5)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_eval_summary(
    metrics_path: Path,
    out_path: Path,
    *,
    algorithm: str,
    env_id: str,
    seed: int,
    train_return_key: str,
) -> dict[str, Any]:
    rows = _read_csv(metrics_path)
    det = [_finite_float(r.get("eval_det_mean")) for r in rows]
    sto = [_finite_float(r.get("eval_sto_mean")) for r in rows]
    train_returns = [_finite_float(r.get(train_return_key)) for r in rows]
    det_vals = [v for v in det if v is not None]
    sto_vals = [v for v in sto if v is not None]
    ret_vals = [v for v in train_returns if v is not None]

    summary = {
        "algorithm": algorithm,
        "env_id": env_id,
        "seed": seed,
        "num_rows": len(rows),
        "best_eval_det_mean": max(det_vals) if det_vals else None,
        "final_eval_det_mean": det_vals[-1] if det_vals else None,
        "best_eval_sto_mean": max(sto_vals) if sto_vals else None,
        "final_eval_sto_mean": sto_vals[-1] if sto_vals else None,
        "final_train_return": ret_vals[-1] if ret_vals else None,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
