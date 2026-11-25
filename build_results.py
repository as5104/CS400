"""Build report tables and figures from the four frozen main experiments."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cs400.comparison import compare_searches, load_search
from cs400.storage import read_json, write_json
from evaluate_main import ARMS, validate_existing


LABELS = ("Random / full", "MOTPE / pruning", "Random / pruning", "MOTPE / full*")
COLORS = ("#555555", "#0072B2", "#009E73", "#D55E00")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = Path("output/final-results")
    output.mkdir(parents=True, exist_ok=True)
    comparison_dir = output / "search-comparison"
    if comparison_dir.exists():
        raise FileExistsError("Results already built; preserve this evidence bundle before rebuilding")
    comparison = compare_searches([Path("runs") / arm for arm in ARMS], comparison_dir)
    models, conditions, epochs, summaries = [], [], [], []
    for arm, search_summary in zip(ARMS, comparison["searches"]):
        directory, search, settings, observations, candidates, _ = load_search(Path("runs") / arm)
        cohorts = dict(candidates)
        for row in search["completed"]:
            cid = row["metrics"]["candidate_id"]
            report_path = directory / "evaluations" / f"{cid}.json"
            validate_existing(report_path, row, search)
            report = read_json(report_path)
            models.append(dict(run=arm, candidate_id=cid, cohort=cohorts[cid], **row["candidate"],
                               validation_accuracy=row["metrics"]["accuracy"],
                               parameters=row["metrics"]["parameters"], training_seconds=row["metrics"]["seconds"],
                               pareto=cid in {r["candidate_id"] for r in search["pareto"]},
                               **report["metrics"],
                               evaluation_seconds=read_json(directory / "evaluation-times" / f"{cid}.json")["evaluation_seconds"]))
            conditions.extend(dict(run=arm, candidate_id=cid, **condition) for condition in report["conditions"])
        for cid, cohort in candidates:
            history = read_json(directory / "trials" / cid / "history.json")
            epochs.extend(dict(run=arm, candidate_id=cid, cohort=cohort, **epoch) for epoch in history)
        arm_models = [m for m in models if m["run"] == arm]
        selected = max(arm_models, key=lambda m: m["validation_accuracy"])
        summaries.append(dict(run=arm, completed=search_summary["completed"],
                              tried=search_summary["proposed_candidates"], pareto_count=search_summary["pareto_count"],
                              hypervolume=search_summary["hypervolume"], active_seconds=search_summary["active_seconds"],
                              budget_seconds=search_summary["active_budget_seconds"],
                              best_validation_accuracy=selected["validation_accuracy"],
                              best_post_startup_validation=max(m["validation_accuracy"] for m in arm_models if m["cohort"] > 0),
                              validation_selected_test_accuracy=selected["clean_accuracy"],
                              validation_selected_corruption_accuracy=selected["corruption_accuracy"],
                              mean_completed_clean_accuracy=statistics.mean(m["clean_accuracy"] for m in arm_models),
                              mean_completed_corruption_accuracy=statistics.mean(m["corruption_accuracy"] for m in arm_models),
                              evaluation_seconds=sum(m["evaluation_seconds"] for m in arm_models)))
    assert len(models) == 29 and len(conditions) == 2175
    distinct = len({m["candidate_id"] for m in models})
    write_csv(output / "models.csv", models)
    write_csv(output / "corruption-conditions.csv", conditions)
    write_csv(output / "epoch-history.csv", epochs)
    write_csv(output / "method-summary.csv", summaries)
    write_json(output / "summary.json", dict(evaluated_checkpoints=len(models), distinct_configurations=distinct,
               corruption_conditions=len(conditions), searches=summaries, analysis="descriptive_only",
               predictor_fitted=False, reason="15 distinct configurations in a single-seed adaptive search sample"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(figure, name):
        figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
        figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for summary, label, color in zip(comparison["searches"], LABELS, COLORS):
        history = summary["hypervolume_history"]
        times = [0] + [r["active_seconds"] / 3600 for r in history] + [summary["active_seconds"] / 3600]
        volumes = [0] + [r["hypervolume"] for r in history] + [summary["hypervolume"]]
        ax.step(times, volumes, where="post", label=label, color=color)
        ax.plot(times[-1], volumes[-1], "o", color=color)
    ax.set(xlabel="Active search time (hours)", ylabel="Validation-objective hypervolume", xlim=(0, 5.25), ylim=(0, .55))
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.text(.12, -.02, "* MOTPE/full includes an operator-reported battery interval. Dots mark search termination.", fontsize=9)
    save(fig, "hypervolume-progress")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        subset = [m for m in models if m["run"] == arm]
        for ax, field, scale in ((axes[0], "parameters", 1e6), (axes[1], "training_seconds", 60)):
            ax.scatter([m[field] / scale for m in subset], [100*m["validation_accuracy"] for m in subset],
                       color=color, label=label, alpha=.7)
            front = [m for m in subset if m["pareto"]]
            ax.scatter([m[field] / scale for m in front], [100*m["validation_accuracy"] for m in front],
                       facecolors="none", edgecolors=color, s=95)
            ax.set_ylabel("Final validation accuracy (%)")
            ax.grid(alpha=.2)
    axes[0].set_xlabel("Parameters (millions)")
    axes[1].set_xlabel("Measured training time (minutes)")
    axes[1].legend(fontsize=8)
    fig.text(.12, -.02, "Rings indicate membership in the three-objective Pareto set. * Includes battery interval.", fontsize=9)
    save(fig, "pareto-projections")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    severity_rows = []
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        values = []
        for severity in range(1, 6):
            rows = [r for r in conditions if r["run"] == arm and r["severity"] == severity]
            mean = statistics.mean(r["accuracy"] for r in rows)
            values.append(100 * mean)
            severity_rows.append(dict(run=arm, severity=severity, mean_accuracy=mean, conditions=len(rows)))
        ax.plot(range(1, 6), values, marker="o", label=label, color=color)
    ax.set(xlabel="Corruption severity", ylabel="Mean corrupted accuracy (%)", xticks=range(1, 6))
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.text(.12, -.02, "Averages cover each method's completed models; these selected model sets differ.", fontsize=9)
    save(fig, "corruption-severity")
    write_csv(output / "severity-summary.csv", severity_rows)

    reference = "13728d774901f951df300fc0d83f24e0d75a3a2763b03243191e654458fa671f"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history = [r for r in epochs if r["run"] == ARMS[0] and r["candidate_id"] == reference]
    axes[0].plot([r["epoch"] for r in history], [r["training_loss"] for r in history], color="black")
    axes[1].plot([r["epoch"] for r in history], [100*r["validation_accuracy"] for r in history], color="black")
    axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    fig.suptitle("Shared reference configuration in random search without pruning")
    fig.tight_layout()
    save(fig, "reference-learning-curve")
    lines = ["# Verified main experimental results", "", "All four searches are frozen. All 29 eligible completed checkpoints have clean-test evaluation and 75 corruption/severity results each.", "",
             "| Search | Tried | Completed | Pareto | Hypervolume | Active hours | Mean clean accuracy | Mean corrupted accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, label in zip(summaries, LABELS):
        lines.append(f"| {label} | {s['tried']} | {s['completed']} | {s['pareto_count']} | {s['hypervolume']:.4f} | {s['active_seconds']/3600:.2f} | {100*s['mean_completed_clean_accuracy']:.2f}% | {100*s['mean_completed_corruption_accuracy']:.2f}% |")
    lines.extend(["", "## Interpretation", "",
        "The shared reference model supplies the highest final validation accuracy in every arm (92.12%). It reaches 91.47% clean test accuracy and 71.584% average corrupted accuracy in all four evaluations. Later proposals did not improve its final validation accuracy in this experiment.", "",
        "MOTPE with pruning has the largest observed hypervolume, using the fixed three-objective reference and actual recorded training times. Actual active-time use differs across arms despite the common 5.25-hour allocation. These single-seed findings describe this experiment.", "",
        "Mean robustness values describe each method's selected archive. The archives have different configurations and different sizes, so these means alone do not establish a causal advantage for a method or optimizer.", "",
        "## Measurement limitations", "",
        "MOTPE without pruning includes a reported battery interval covering reference-model epochs 36–60 and the following candidate's epochs 1–17. Epoch 17 is the reported transition. The saved durations are retained. This affects time objectives, admission decisions and hypervolume comparability. See ../main-motpe-full-power-note.md.", "",
        "There are 15 distinct configurations among the 29 checkpoints. Per-run measurements are retained because a configuration identifier does not uniquely identify a trained checkpoint. One startup configuration has 91.00% versus 91.01% clean accuracy across runs. A checkpoint comparison found weight differences, with training-metric differences beginning at epoch 51 for the checked pair. The cause is unresolved. The six-epoch reference resume test passed exactly, but does not prove exact reproducibility for every configuration and duration.", "",
        "Robustness characterization remains descriptive with this limited configuration sample. No random-forest predictor or feature-importance result is fitted. Repeated configurations are not independent random-seed replications.", "",
        "## Artifacts", "",
        "models.csv joins configuration, validation, clean-test, corruption and timing measurements by run and configuration. corruption-conditions.csv contains all 2,175 per-condition rows. epoch-history.csv includes completed and pruned candidate histories. method-summary.csv and severity-summary.csv contain the plotted summaries. Plots are available as PNG and PDF. Data fractions in CSV/JSON become percentages only in displayed tables and axes.", "",
        f"Measured active search time totals {sum(s['active_seconds'] for s in summaries)/3600:.2f} hours. Recorded evaluation time totals {sum(s['evaluation_seconds'] for s in summaries)/60:.2f} minutes. These exclude earlier pilots, integration tests and intentional pauses."])
    (output / "Results_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), models=len(models), distinct_configurations=distinct,
                          conditions=len(conditions), epoch_rows=len(epochs), figures=4), indent=2))


if __name__ == "__main__":
    main()
