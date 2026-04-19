import csv
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
