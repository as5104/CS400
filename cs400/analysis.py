"""Optional post-search prediction, guarded by configuration-level data volume."""
from pathlib import Path
import math
import numpy as np
from .config import Candidate
from .storage import read_json, write_json
from .evaluation import robustness_metrics


def analyze_reports(paths, output):
    if Path(output).exists():
        raise FileExistsError("Analysis output exists; choose a new path")
    reports = {}
    for path in paths:
        report = read_json(path)
        if report.get("phase") != "post_search" or len(report.get("conditions", [])) != 75:
            raise ValueError("Analysis requires completed post-search corruption reports")
        metrics = robustness_metrics(report["metrics"]["clean_accuracy"],
                                     {(r["corruption"], r["severity"]): r["accuracy"] for r in report["conditions"]})
        if abs(metrics["mean_corruption_error"] - report["metrics"]["mean_corruption_error"]) > 1e-10:
            raise ValueError("Aggregate corruption metric disagrees with per-condition records")
        # Configuration/environment identity does not uniquely identify trained weights.
        # Legacy reports have no artifact hash: retain separate files as separate observations.
        key = report.get("artifact_id") or str(Path(path).resolve())
        if key in reports and report != reports[key]:
            raise ValueError("Conflicting evaluations for the same trained model")
        reports[key] = report
    rows = list(reports.values())
    if not rows:
        raise ValueError("No evaluation reports supplied")
    if len({(r["manifest_id"], r["epoch"]) for r in rows}) != 1:
        raise ValueError("Do not mix datasets or full training budgets in one analysis")
    groups = np.asarray([Candidate(**r["candidate"]).id for r in rows])
    distinct = len(set(groups))
    result = dict(trained_models=len(rows), distinct_configurations=distinct, causal_claims=False,
                  legacy_identity_note="Legacy reports are deduplicated by file path; repeated configurations remain grouped",
                  target="mean_corruption_error", sampling_limitation="Adaptive search and pruning select this sample; findings concern observed configurations")
    if distinct < 40:
        result.update(status="descriptive_only", reason="Fewer than 40 distinct configurations; no predictor fitted",
                      mean_corruption_error=float(np.mean([r["metrics"]["mean_corruption_error"] for r in rows])))
        write_json(output, result)
        return result
    # Optional dependency, imported only when the user explicitly runs analysis.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    names = ["depth", "optimizer", "augmentation", "width", "log_learning_rate", "log_weight_decay", "dropout"]
    features = []
    for row in rows:
        c = Candidate(**row["candidate"])
        features.append([c.depth, c.optimizer, c.augmentation, c.width, math.log10(c.learning_rate), math.log10(c.weight_decay), c.dropout])
    X = np.asarray(features, dtype=object)
    y = np.asarray([r["metrics"]["mean_corruption_error"] for r in rows])
    clean = np.asarray([r["metrics"]["clean_accuracy"] for r in rows]).reshape(-1, 1)
    predictions, baseline, clean_prediction = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    importances = []
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        encoder = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1, 2]),
                                     ("numeric", "passthrough", [3, 4, 5, 6])])
        predictor = make_pipeline(encoder, RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                                                random_state=42, n_jobs=1))
        predictor.fit(X[train], y[train])
        predictions[test] = predictor.predict(X[test])
        baseline[test] = DummyRegressor().fit(X[train], y[train]).predict(X[test])
        clean_prediction[test] = Ridge().fit(clean[train], y[train]).predict(clean[test])
        importances.append(permutation_importance(predictor, X[test], y[test], scoring="neg_mean_absolute_error",
                                                   n_repeats=10, random_state=42).importances_mean)
    mae = float(mean_absolute_error(y, predictions))
    baseline_mae = float(mean_absolute_error(y, baseline))
    r2 = float(r2_score(y, predictions))
    result.update(status="exploratory" if distinct < 80 else "internally_cross_validated", folds=5,
                  mae=mae, rmse=float(math.sqrt(mean_squared_error(y, predictions))), r_squared=r2,
                  mean_baseline_mae=baseline_mae, clean_accuracy_only_mae=float(mean_absolute_error(y, clean_prediction)),
                  interpretation_supported=mae < baseline_mae and r2 > 0)
    if result["interpretation_supported"]:
        result["held_out_permutation_importance"] = dict(zip(names, np.mean(importances, axis=0).tolist()))
    write_json(output, result)
    return result
