"""Tests for sepsis_mining.data against a tiny fake export."""

import hashlib
import json

import polars as pl
import pytest

from sepsis_mining.data import latest_export, scan_mart, split_filter, verify_manifest


@pytest.fixture
def export(tmp_path):
    d = tmp_path / "v1_20261101"
    rows = {
        "train": [(1, 0, "train", "train", 0), (1, 1, "train", "validation", 1)],
        "test": [(2, 0, "test", "test", 2), (3, 0, "test", "train", 2)],
    }
    cols = ["patient_id", "iculos", "meta_split_random", "meta_split_hospital_holdout",
            "meta_cv_fold"]  # fmt: skip
    for split, r in rows.items():
        part = d / "ml_features_patient_hour" / f"meta_split_random={split}"
        part.mkdir(parents=True)
        pl.DataFrame(r, schema=cols, orient="row").write_parquet(part / "0.parquet")
    (d / "manifest.json").write_text(json.dumps({"rows_per_split": {"train": 2, "test": 2}}))
    return d


def test_scan_mart_reads_all_partitions_without_duplicate_columns(export):
    df = scan_mart(export).collect()
    assert df.height == 4
    assert len(df.columns) == len(set(df.columns)) == 5


def test_split_filter_selects_rows(export):
    df = scan_mart(export).collect()
    assert df.filter(split_filter("random", "test"))["patient_id"].to_list() == [2, 3]
    assert df.filter(split_filter("hospital_holdout", "validation"))["patient_id"].to_list() == [1]
    assert df.filter(split_filter("cv", "fold_2"))["patient_id"].to_list() == [2, 3]


def test_verify_manifest_returns_data_version(export):
    version = verify_manifest(export)
    digest = hashlib.sha256((export / "manifest.json").read_bytes()).hexdigest()[:12]
    assert version == f"v1_20261101+{digest}"


def test_verify_manifest_names_mismatching_split(export):
    (export / "manifest.json").write_text(json.dumps({"rows_per_split": {"train": 2, "test": 9}}))
    with pytest.raises(ValueError, match="test"):
        verify_manifest(export)


def test_latest_export_picks_newest(tmp_path):
    (tmp_path / "v1_20261001").mkdir()
    (tmp_path / "v1_20261101").mkdir()
    assert latest_export(tmp_path).name == "v1_20261101"


def test_latest_export_empty_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_export(tmp_path)
