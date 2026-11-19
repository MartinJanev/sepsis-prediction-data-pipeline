"""Load the Parquet export (read-only; the database is never touched)."""

import hashlib
import json
from pathlib import Path

import polars as pl

SCHEME_COLUMN = {
    "random": "meta_split_random",
    "hospital_holdout": "meta_split_hospital_holdout",
}


def latest_export(root: Path = Path("exports")) -> Path:
    """Newest `v*` export directory under `root` (by name)."""
    dirs = sorted(p for p in root.glob("v*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no export directory v* under {root}")
    return dirs[-1]


def scan_mart(export_dir: Path) -> pl.LazyFrame:
    """All partitions of the feature mart. Files already hold the partition column."""
    return pl.scan_parquet(
        export_dir / "ml_features_patient_hour" / "**" / "*.parquet", hive_partitioning=False
    )


def split_filter(scheme: str, split: str) -> pl.Expr:
    """Row filter for a split; for `cv`, `split` is `fold_<k>`."""
    if scheme == "cv":
        return pl.col("meta_cv_fold") == int(split.removeprefix("fold_"))
    return pl.col(SCHEME_COLUMN[scheme]) == split


def verify_manifest(export_dir: Path) -> str:
    """Check manifest row counts against the data; return the data-version string."""
    raw = (export_dir / "manifest.json").read_bytes()
    expected = json.loads(raw)["rows_per_split"]
    counts = scan_mart(export_dir).group_by("meta_split_random").len().collect()
    actual = dict(zip(counts["meta_split_random"], counts["len"], strict=True))
    for split, n in expected.items():
        if actual.get(split, 0) != n:
            raise ValueError(
                f"split {split!r}: manifest says {n} rows, found {actual.get(split, 0)}"
            )
    return f"{export_dir.name}+{hashlib.sha256(raw).hexdigest()[:12]}"
