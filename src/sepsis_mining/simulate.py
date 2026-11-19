"""Synthetic rehearsal: simulate a cohort with known hazard coefficients, then recover them."""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

BETA = {"feat_x1": 0.8, "feat_x2": 0.5, "feat_age": 0.03}
_PHI = 0.8
_MAX_LOS = 72
_MAX_TAIL = 10


def _inv_link(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        return 1 / (1 + np.exp(-eta))
    return 1 - np.exp(-np.exp(eta))  # cloglog


def simulate_cohort(
    n_patients: int, seed: int, beta: dict, alpha: float = -5.0, link: str = "logit"
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    h_len = _MAX_LOS + _MAX_TAIL
    age = rng.normal(60, 15, n_patients)
    ar = {}
    for name in ("feat_x1", "feat_x2"):
        x = np.empty((n_patients, h_len))
        x[:, 0] = rng.standard_normal(n_patients)
        noise = rng.standard_normal((n_patients, h_len)) * np.sqrt(1 - _PHI**2)
        for t in range(1, h_len):
            x[:, t] = _PHI * x[:, t - 1] + noise[:, t]
        ar[name] = x
    cols = {"feat_x1": ar["feat_x1"], "feat_x2": ar["feat_x2"], "feat_age": age[:, None]}
    eta = alpha + sum(beta[k] * cols[k] for k in beta)  # (n, h_len) via broadcasting
    eta = np.broadcast_to(eta, (n_patients, h_len))
    events = rng.random((n_patients, h_len)) < _inv_link(eta, link)
    los = rng.integers(8, _MAX_LOS + 1, n_patients)
    tail = rng.integers(0, _MAX_TAIL + 1, n_patients)
    hours = np.arange(1, h_len + 1)
    events &= hours <= los[:, None]
    septic = events.any(axis=1)
    onset = np.where(septic, events.argmax(axis=1) + 1, 0)  # iculos of T_i
    length = np.where(septic, onset + tail, los)
    keep = hours[None, :] <= length[:, None]
    pid, col = np.nonzero(keep)
    iculos = col + 1
    return pl.DataFrame(
        {
            "patient_id": pid.astype(np.int64),
            "iculos": iculos.astype(np.int64),
            "label_sepsis": (septic[pid] & (iculos >= onset[pid])).astype(np.int64),
            "feat_x1": ar["feat_x1"][pid, col],
            "feat_x2": ar["feat_x2"][pid, col],
            "feat_age": age[pid],
        }
    )


def recovery_table(coefs: pl.DataFrame, alpha: float, beta: dict) -> pl.DataFrame:
    true = pl.when(pl.col("term").str.starts_with("alpha[")).then(alpha)
    for k, v in beta.items():
        true = true.when(pl.col("term") == k).then(v)
    return coefs.select(
        "term",
        true.alias("true"),
        "estimate",
        "ci_low",
        "ci_high",
    ).with_columns(
        ((pl.col("ci_low") <= pl.col("true")) & (pl.col("true") <= pl.col("ci_high"))).alias(
            "covered"
        )
    )


def _plot(table: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    t, e = table["true"].to_numpy(), table["estimate"].to_numpy()
    err = np.vstack([e - table["ci_low"].to_numpy(), table["ci_high"].to_numpy() - e])
    ax.errorbar(t, e, yerr=err, fmt="o", capsize=3)
    lim = [min(t.min(), e.min()) - 0.5, max(t.max(), e.max()) + 0.5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="true", ylabel="estimate (95% CI)", title="Hazard coefficient recovery")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from sepsis_mining.hazard import fit_hazard
    from sepsis_mining.targets import add_targets

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/simulation"))
    p.add_argument("--n", type=int, default=5000)
    args = p.parse_args()
    frame = add_targets(simulate_cohort(args.n, 0, BETA))
    table = recovery_table(fit_hazard(frame, list(BETA)), -5.0, BETA)
    args.out.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out / "recovery.csv")
    _plot(table, args.out / "recovery.png")
    with pl.Config(tbl_rows=20):
        print(table)


if __name__ == "__main__":
    main()
