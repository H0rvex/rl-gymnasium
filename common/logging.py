import csv
import math
from pathlib import Path
from typing import Any


class CsvLogger:
    """Append-only CSV writer with schema validation and per-row flush.

    Usage:
        with CsvLogger(path, fieldnames=[...]) as logger:
            logger.log({"episode": 0, "ep_return": 123.0, ...})

    Missing columns are filled with NaN. Unknown columns raise — this catches
    typos in row dicts early, which is the whole point of a validated logger.
    """

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._f: Any = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "CsvLogger":
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
            self._writer = None

    def log(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("CsvLogger used outside of its `with` block")
        extra = set(row) - set(self.fieldnames)
        if extra:
            raise ValueError(f"unknown CSV fields: {sorted(extra)}")
        filled = {k: row.get(k, float("nan")) for k in self.fieldnames}
        self._writer.writerow(filled)
        self._f.flush()


class TbLogger:
    """Thin context-manager wrapper around torch.utils.tensorboard.SummaryWriter.

    Matches CsvLogger's lifecycle so both can be opened in a single `with` block.
    NaN values are silently skipped — TensorBoard rejects them and they carry no
    information worth plotting.

    Usage:
        with TbLogger(log_dir) as tb:
            tb.scalar("train/loss", loss, step=episode)
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self._writer: Any = None

    def __enter__(self) -> "TbLogger":
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(self.log_dir))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            raise RuntimeError("TbLogger used outside of its `with` block")
        if not math.isnan(value):
            self._writer.add_scalar(tag, value, global_step=step)
