"""Discrete-time hazard fit: binomial GLM on the at-risk person-period rows."""

import numpy as np
import polars as pl
import statsmodels.api as sm

GROUPS = ["0-6", "7-12", "13-24", "25-48", ">48"]
_FAMILIES = {"logit": sm.families.links.Logit, "cloglog": sm.families.links.CLogLog}


def fit_hazard(
    frame: pl.DataFrame | pl.LazyFrame, features: list[str], link: str = "logit"
) -> pl.DataFrame:
    """One alpha per ICU-time group (no global intercept) + `features`; 95% Wald CIs."""
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    risk = frame.filter(pl.col("label_hazard_at_risk"))
    missing = [g for g in GROUPS if g not in set(risk["feat_icu_time_group"])]
    if missing:
        raise ValueError(f"no at-risk rows in ICU-time group(s): {missing}")
    names = [f"alpha[{g}]" for g in GROUPS]
    alphas = np.column_stack([(risk["feat_icu_time_group"] == g).to_numpy() for g in GROUPS])
    x = np.column_stack([alphas.astype(float), risk.select(features).to_numpy()])
    y = risk["label_hazard_event"].to_numpy().astype(float)
    fit = sm.GLM(y, x, family=sm.families.Binomial(link=_FAMILIES[link]())).fit()
    est, se = fit.params, fit.bse
    lo, hi = fit.conf_int().T
    return pl.DataFrame(
        {
            "term": names + features,
            "estimate": est,
            "std_error": se,
            "odds_ratio": np.exp(est) if link == "logit" else [None] * len(est),
            "ci_low": lo,
            "ci_high": hi,
        },
        schema_overrides={"odds_ratio": pl.Float64},
    )
