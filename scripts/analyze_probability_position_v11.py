"""Post-hoc WDBC v1.1 diagnostics, computed ONLY from frozen public summaries.

This file neither refits a classifier nor overwrites any primary artifact.
Run from any location: python scripts/analyze_probability_position_v11.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "results"
OUTPUT = ROOT / "analysis" / "generated_v11"

MODELS = {
    "logistic_regression": SOURCE / "logistic_regression_individual_stability.csv",
    "random_forest": SOURCE / "random_forest_individual_stability.csv",
}
CENTRAL_THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
ROLLING_WINDOWS = (31, 51, 71)


def load_and_validate(source: Path = SOURCE) -> dict[str, pd.DataFrame]:
    tables = {}
    cols = {"observation_id", "y_true", "median_probability", "resampling_width", "n_oos_predictions"}
    for model in MODELS:
        path = source / f"{model}_individual_stability.csv"
        table = pd.read_csv(path)
        missing = cols - set(table.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if table.observation_id.duplicated().any():
            raise ValueError(f"{path}: duplicate observation_id")
        if not table.median_probability.between(0, 1).all():
            raise ValueError(f"{path}: out-of-range probability")
        if not table.resampling_width.between(0, 1).all():
            raise ValueError(f"{path}: invalid width")
        if not (table.n_oos_predictions == 50).all():
            raise ValueError(f"{path}: unexpected n_oos_predictions")
        if not table.y_true.isin([0, 1]).all():
            raise ValueError(f"{path}: invalid diagnosis")
        table = table.sort_values("observation_id").reset_index(drop=True)
        table["position_geometry"] = table.median_probability * (1 - table.median_probability)
        tables[model] = table
    a, b = tables.values()
    if len(a) != 569 or len(b) != 569:
        raise ValueError("Frozen WDBC experiment requires 569 observations per model")
    if not a.observation_id.equals(b.observation_id) or not a.y_true.equals(b.y_true):
        raise ValueError("Misaligned observations or outcome coding between models")
    return tables


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def residualize_rank(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of rank(y) ~ intercept + rank(x); no statistical inference."""
    yy = rankdata(y, method="average")
    xx = rankdata(x, method="average")
    coef, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(xx)), xx)), yy, rcond=None)
    return yy - (coef[0] + coef[1] * xx)


def rolling_excess(table: pd.DataFrame, window: int) -> pd.DataFrame:
    """Exploratory, rank-neighborhood baseline; neither fitted uncertainty nor a test."""
    sorted_table = table.sort_values(["position_geometry", "observation_id"]).copy()
    baseline = sorted_table.resampling_width.rolling(window=window, center=True, min_periods=1).median()
    sorted_table["local_median_width"] = baseline
    sorted_table["excess_width"] = sorted_table.resampling_width - baseline
    sorted_table["window"] = window
    return sorted_table


def analyze(tables: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    lr, rf = tables["logistic_regression"], tables["random_forest"]
    wa = lr.resampling_width.to_numpy()
    wb = rf.resampling_width.to_numpy()
    ga = lr.position_geometry.to_numpy()
    gb = rf.position_geometry.to_numpy()
    result = {
        "analysis_version": "1.1.0",
        "status": "post_hoc_exploratory",
        "sample_n": len(lr),
        "metrics": {
            "spearman_lr_rf_width": spearman(wa, wb),
            "spearman_lr_width_vs_p_one_minus_p": spearman(wa, ga),
            "spearman_rf_width_vs_p_one_minus_p": spearman(wb, gb),
            "pearson_lr_width_vs_p_one_minus_p": pearson(wa, ga),
            "pearson_rf_width_vs_p_one_minus_p": pearson(wb, gb),
            "rank_residual_correlation": pearson(residualize_rank(wa, ga), residualize_rank(wb, gb)),
        },
        "central_sensitivity": [],
        "rolling_window_sensitivity": {},
    }
    for lo in CENTRAL_THRESHOLDS:
        mask = lr.median_probability.between(lo, 1 - lo) & rf.median_probability.between(lo, 1 - lo)
        n = int(mask.sum())
        result["central_sensitivity"].append({
            "minimum_median_probability": lo,
            "maximum_median_probability": 1 - lo,
            "n": n,
            "spearman_lr_rf_width": spearman(wa[mask.to_numpy()], wb[mask.to_numpy()]) if n > 2 else None,
        })

    candidates = []
    for model, table in tables.items():
        tops = {}
        for window in ROLLING_WINDOWS:
            residuals = rolling_excess(table, window)
            residuals = residuals.sort_values(["excess_width", "observation_id"], ascending=[False, True])
            tops[str(window)] = residuals.observation_id.head(10).tolist()
            if window == 51:
                candidates.append(residuals.head(10).assign(model=model)[[
                    "model", "observation_id", "y_true", "median_probability", "position_geometry",
                    "resampling_width", "local_median_width", "excess_width", "window"
                ]])
        stable = sorted(set(tops["31"]) & set(tops["51"]) & set(tops["71"]))
        result["rolling_window_sensitivity"][model] = {
            "top10_ids_common_to_windows_31_51_71": stable,
            "top10_ids_window_51": tops["51"],
        }
    return result, pd.concat(candidates, ignore_index=True)


def make_figure(tables: dict[str, pd.DataFrame], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
    for ax, (model, table) in zip(axes, tables.items()):
        for diagnosis, label in [(0, "Benign"), (1, "Malignant")]:
            subset = table.loc[table.y_true == diagnosis]
            ax.scatter(subset.position_geometry, subset.resampling_width, s=13, alpha=0.6,
                       label=label, rasterized=True)
        ordered = rolling_excess(table, 51)
        ax.plot(ordered.position_geometry, ordered.local_median_width,
                color="black", lw=1.8, label="51-case rolling median")
        ax.set_title("Logistic regression" if model == "logistic_regression" else "Random forest")
        ax.set_xlabel("Median p × (1 − median p)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Resampling width W (P90 − P10)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Exploratory position–width relationship (WDBC v1.1)")
    fig.savefig(destination / "probability_position_v11.png", dpi=180)
    fig.savefig(destination / "probability_position_v11.pdf")
    plt.close(fig)


def main() -> None:
    tables = load_and_validate()
    metrics, candidates = analyze(tables)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "position_diagnostics_v11.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    candidates.to_csv(OUTPUT / "positive_excess_candidates_v11.csv", index=False, float_format="%.15g")
    make_figure(tables, OUTPUT)
    print(json.dumps({"metrics": metrics["metrics"], "central_sensitivity": metrics["central_sensitivity"]}, indent=2))
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
