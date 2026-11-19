import json

import numpy as np
import polars as pl
import pytest

from sepsis_mining.evaluate import (
    alarm_timing,
    choose_threshold,
    classification_metrics,
    normalised_utility,
    prediction_frame,
    validate_predictions,
    write_run_record,
)
from sepsis_mining.targets import add_targets

NORMS = pl.DataFrame(
    {
        "split_scheme": ["s"],
        "split": ["val"],
        "best_utility_total": [2.0],
        "inaction_utility_total": [-1.0],
        "n_rows": [3],
    }
)


def _util_frame(pos, neg):
    return pl.DataFrame({"label_utility_if_positive": pos, "label_utility_if_negative": neg})


def test_normalised_utility_oracle_and_inaction():
    f = _util_frame([1.0, -0.05, 0.5], [0.0, 0.0, -1.0])
    assert normalised_utility(f, np.array([1, 0, 1]), NORMS, "s", "val") == pytest.approx(2.5 / 3)
    zero = normalised_utility(f, np.zeros(3), NORMS, "s", "val")
    assert zero == pytest.approx((-1.0 + 1.0) / 3)


def test_choose_threshold_excludes_harmful_top_row():
    f = _util_frame([-1.0, -0.5, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0])
    risk = np.array([0.9, 0.5, 0.4, 0.1])
    norms = NORMS.with_columns(n_rows=4)
    theta = choose_threshold(f, risk, norms, "s", "val")
    assert not (risk[0] >= theta)  # top-risk (negative utility) row not alarmed
    best = max(
        normalised_utility(f, (risk >= t).astype(int), norms, "s", "val") for t in [*risk, 1.0]
    )
    assert normalised_utility(f, (risk >= theta).astype(int), norms, "s", "val") == pytest.approx(
        best
    )


def test_normalised_utility_rejects_subset_and_bad_keys():
    f = _util_frame([1.0, -0.05], [0.0, 0.0])
    with pytest.raises(ValueError, match="n_rows"):
        normalised_utility(f, np.array([1, 0]), NORMS, "s", "val")
    f3 = _util_frame([1.0, -0.05, 0.5], [0.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="alarm"):
        normalised_utility(f3, np.array([1, 0]), NORMS, "s", "val")
    with pytest.raises(ValueError, match="cv_fold_2"):
        normalised_utility(f3, np.zeros(3), NORMS, "cv_fold_2", "val")
    with pytest.raises(ValueError, match="n_rows"):
        choose_threshold(f, np.array([0.1, 0.2]), NORMS, "s", "val")
    with pytest.raises(ValueError, match="risk"):
        choose_threshold(f3, np.array([0.1, 0.2]), NORMS, "s", "val")


def test_classification_metrics_oracle():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["auroc"] == 0.75
    assert m["brier"] == pytest.approx(0.158125)
    assert "auprc" in m


def test_alarm_timing_oracle(mini_mart):
    f = add_targets(mini_mart)
    alarms = {("p_sep", 5), ("p_sep", 6), ("p_left", 0), ("p_non", 1), ("p_non", 2)}
    on = [(p, r) in alarms for p, r in zip(f["patient_id"], f["row_idx"], strict=True)]
    out = alarm_timing(f.with_columns(pl.Series("binary_alarm", on).cast(pl.Int8)))
    assert out["first_alarm_minus_onset_median"] == -1.5
    assert out["share_alarmed_in_window"] == pytest.approx(2 / 3)
    assert out["false_alarms_per_patient_day"] == pytest.approx(4.8)
    assert len(out["first_alarm_minus_onset_iqr"]) == 2


def _pred(mini_mart):
    f = add_targets(mini_mart)
    risk = np.full(f.height, 0.3)
    return f, prediction_frame(
        f,
        risk,
        0.5,
        model_name="m",
        feature_set="fs",
        target="y6",
        split="val",
        split_scheme="s",
        run_id="r1",
    )


def test_prediction_frame_columns(mini_mart):
    _, p = _pred(mini_mart)
    assert p.columns == [
        "patient_id", "hour", "row_idx", "true_label", "target", "predicted_risk",
        "binary_alarm", "model_name", "feature_set", "threshold", "split", "split_scheme",
        "run_id",
    ]  # fmt: skip
    assert p["binary_alarm"].sum() == 0


def _first_row(value):
    return pl.when(pl.int_range(pl.len()) == 0).then(value).otherwise(0.3)


def test_validate_predictions(mini_mart):
    f, p = _pred(mini_mart)
    keys = f.select("patient_id", pl.col("iculos").alias("hour"))
    validate_predictions(p, keys)
    cases = {
        "missing": p.slice(1),
        "duplicate": pl.concat([p, p.head(1)]),
        "outside": p.with_columns(predicted_risk=_first_row(1.2)),
        "null": p.with_columns(predicted_risk=_first_row(None)),
        "binary_alarm": p.with_columns(binary_alarm=pl.lit(2)),
        "target": p.with_columns(target=pl.lit("foo")),
        "columns": p.drop("run_id"),
    }  # fmt: skip
    for needle, bad in cases.items():
        with pytest.raises(ValueError, match=needle):
            validate_predictions(bad, keys)
    with pytest.raises(ValueError, match="extra"):
        validate_predictions(p, keys.slice(1))
    for col, needle in (("binary_alarm", "binary_alarm"), ("target", "target")):
        bad = p.with_columns(_first_row(None).cast(p[col].dtype).alias(col))
        with pytest.raises(ValueError, match=needle):
            validate_predictions(bad, keys)


def test_write_run_record(tmp_path):
    path = write_run_record(
        tmp_path / "results",
        run_id="r1",
        config={"a": 1},
        data_version="dv",
        seed=7,
        split_scheme="s",
        metrics={"auroc": 0.8},
        artifacts=["x.parquet"],
        author="me",
    )
    rec = json.loads(path.read_text())
    assert path.name.startswith("run_") and path.name.endswith("_r1.json")
    assert rec.keys() == {
        "run_id", "git_commit", "config", "data_version", "seed", "split_scheme",
        "metrics", "artifacts", "started_at", "author",
    }  # fmt: skip
