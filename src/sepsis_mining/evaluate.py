"""Evaluation harness: challenge utility, metrics, alarm timing, prediction/run records."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sepsis_pipeline.audit import current_git_sha

PRED_COLUMNS = [
    "patient_id", "hour", "row_idx", "true_label", "target", "predicted_risk", "binary_alarm",
    "model_name", "feature_set", "threshold", "split", "split_scheme", "run_id",
]  # fmt: skip


def normalised_utility(frame, alarm, norms, scheme, split) -> float:
    """Needs the WHOLE split's rows: totals in `norms` cover every row of (scheme, split)."""
    rows = norms.filter((pl.col("split_scheme") == scheme) & (pl.col("split") == split))
    if rows.height != 1:
        raise ValueError(f"no norms row for split_scheme={scheme!r}, split={split!r}")
    row = rows.row(0, named=True)
    if frame.height != row["n_rows"]:
        raise ValueError(
            f"frame has {frame.height} rows but norms n_rows={row['n_rows']} "
            f"({scheme!r}, {split!r}): pass the whole split, not a subset"
        )
    if len(alarm) != frame.height:
        raise ValueError(f"alarm length {len(alarm)} != frame rows {frame.height}")
    observed = np.where(
        np.asarray(alarm) == 1,
        frame["label_utility_if_positive"].to_numpy(),
        frame["label_utility_if_negative"].to_numpy(),
    ).sum()
    inaction, best = row["inaction_utility_total"], row["best_utility_total"]
    return float((observed - inaction) / (best - inaction))


def classification_metrics(y, risk) -> dict:
    return {
        "auroc": float(roc_auc_score(y, risk)),
        "auprc": float(average_precision_score(y, risk)),
        "brier": float(brier_score_loss(y, risk)),
    }


def choose_threshold(frame, risk, norms, scheme, split, n_candidates: int = 200) -> float:
    """Risk-quantile θ maximising utility (validation only). Includes 'never alarm'."""
    risk = np.asarray(risk)
    if len(risk) != frame.height:
        raise ValueError(f"risk length {len(risk)} != frame rows {frame.height}")
    cands = [*np.unique(np.quantile(risk, np.linspace(0, 1, n_candidates))), np.inf]
    scores = [
        normalised_utility(frame, (risk >= t).astype(int), norms, scheme, split) for t in cands
    ]
    return float(cands[int(np.argmax(scores))])


def alarm_timing(pred: pl.DataFrame) -> dict:
    """Needs patient_id, row_idx, label_onset_row_idx, binary_alarm. Alarm hours, no latching.

    Works on preprocess output: pass `pre.with_columns(pl.Series("binary_alarm", alarms))`
    with `alarms` an int 0/1 array.
    """
    septic = pred.filter(pl.col("label_onset_row_idx").is_not_null())
    alarmed = septic.filter(pl.col("binary_alarm") == 1)
    delta = alarmed.group_by("patient_id").agg(
        (pl.col("row_idx").min() - pl.col("label_onset_row_idx").first()).alias("d")
    )["d"]
    rel = pl.col("row_idx") - pl.col("label_onset_row_idx")
    in_window = alarmed.filter(rel.is_between(-6, 9))["patient_id"].n_unique()
    n_septic = septic["patient_id"].n_unique()
    non = pred.filter(pl.col("label_onset_row_idx").is_null())
    return {
        "first_alarm_minus_onset_median": delta.median(),
        "first_alarm_minus_onset_iqr": (delta.quantile(0.25), delta.quantile(0.75)),
        "share_alarmed_in_window": in_window / n_septic if n_septic else None,
        "false_alarms_per_patient_day": (
            (non["binary_alarm"] == 1).sum() / (non.height / 24) if non.height else None
        ),
    }


def prediction_frame(
    frame, risk, threshold, *, model_name, feature_set, target, split, split_scheme, run_id
) -> pl.DataFrame:
    risk = np.asarray(risk, dtype=float)
    return frame.select(
        "patient_id",
        pl.col("iculos").alias("hour"),
        "row_idx",
        pl.col("label_sepsis").alias("true_label"),
    ).with_columns(
        pl.lit(target).alias("target"),
        pl.Series("predicted_risk", risk),
        pl.Series("binary_alarm", (risk >= threshold).astype(int)),
        pl.lit(model_name).alias("model_name"),
        pl.lit(feature_set).alias("feature_set"),
        pl.lit(float(threshold)).alias("threshold"),
        pl.lit(split).alias("split"),
        pl.lit(split_scheme).alias("split_scheme"),
        pl.lit(run_id).alias("run_id"),
    )


def validate_predictions(pred: pl.DataFrame, expected_keys: pl.DataFrame) -> None:
    missing_cols = [c for c in PRED_COLUMNS if c not in pred.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    keys = ["patient_id", "hour"]
    problems = []
    got, want = pred.select(keys), expected_keys.select(keys)
    if (n := got.is_duplicated().sum()) > 0:
        problems.append(f"{n} duplicate (patient_id, hour) rows")
    if (n := want.join(got, on=keys, how="anti").height) > 0:
        problems.append(f"{n} missing (patient_id, hour) keys")
    if (n := got.unique().join(want, on=keys, how="anti").height) > 0:
        problems.append(f"{n} extra (patient_id, hour) keys")
    risk = pred["predicted_risk"]
    if risk.null_count() or risk.is_nan().any():
        problems.append("null predicted_risk")
    if (~risk.is_between(0, 1)).any():
        problems.append("predicted_risk outside [0, 1]")
    if (~pred["binary_alarm"].is_in([0, 1]).fill_null(False)).any():
        problems.append("binary_alarm not in {0, 1}")
    if (~pred["target"].is_in(["y6", "hazard"]).fill_null(False)).any():
        problems.append('target not in {"y6", "hazard"}')
    if problems:
        raise ValueError("; ".join(problems))


def write_run_record(
    results_dir, *, run_id, config, data_version, seed, split_scheme, metrics, artifacts, author
) -> Path:
    now = datetime.now()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"run_{now:%Y%m%dT%H%M%S}_{run_id}.json"
    record = {
        "run_id": run_id,
        "started_at": now.isoformat(timespec="seconds"),
        "git_commit": current_git_sha(),
        "config": config,
        "data_version": data_version,
        "seed": seed,
        "split_scheme": split_scheme,
        "metrics": metrics,
        "artifacts": [str(a) for a in artifacts],
        "author": author,
    }
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
