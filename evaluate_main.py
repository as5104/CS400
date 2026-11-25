"""Evaluate frozen main-search checkpoints sequentially, in optional batches."""
import argparse
import math
import time
from pathlib import Path

from cs400.comparison import load_search
from cs400.config import digest
from cs400.evaluation import evaluate_run, robustness_metrics
from cs400.storage import read_json, write_json


ARMS = ("main-random-full", "main-motpe-halving", "main-random-halving", "main-motpe-full")


def validate_existing(path, row, search):
    report = read_json(path)
    metadata = read_json(Path(row["run_dir"]) / "run.json")
    if (report["phase"] != "post_search" or report["training_identity"] != digest(metadata)
            or report["candidate"] != row["candidate"] or report["epoch"] != row["metrics"]["epoch"]
            or report["manifest_id"] != search["manifest_id"]):
        raise ValueError(f"Evaluation identity mismatch in {path}")
    rows = report["conditions"]
    if len(rows) != 75 or any(r["samples"] != 10000 for r in rows):
        raise ValueError(f"Incomplete evaluation in {path}")
    scores = {(r["corruption"], r["severity"]): r["accuracy"] for r in rows}
    metrics = robustness_metrics(report["metrics"]["clean_accuracy"], scores)
    if any(not math.isclose(v, report["metrics"][k], abs_tol=1e-10) for k, v in metrics.items()):
        raise ValueError(f"Inconsistent metrics in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Maximum number of new models to evaluate this session")
    parser.add_argument("--list", action="store_true", help="Count remaining evaluations without using the GPU")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    pending = []
    total = 0
    for arm in ARMS:
        directory, search, *_ = load_search(Path("runs") / arm)
        for row in search["completed"]:
            total += 1
            cid = row["metrics"]["candidate_id"]
            path = directory / "evaluations" / f"{cid}.json"
            if path.exists():
                validate_existing(path, row, search)
            else:
                pending.append((arm, row, search, path))
    print(f"{total - len(pending)}/{total} reports complete; {len(pending)} remaining.", flush=True)
    if args.list:
        return
    for i, (arm, row, search, path) in enumerate(pending[:args.limit], 1):
        print(f"Evaluating {arm} / {path.stem[:12]} (model {i} this session)...", flush=True)
        started = time.perf_counter()
        evaluate_run(row["run_dir"], "data/cifar10-split.json", "data", "data/CIFAR-10-C", path)
        elapsed = time.perf_counter() - started
        validate_existing(path, row, search)
        write_json(path.parent.parent / "evaluation-times" / path.name,
                   dict(candidate_id=path.stem, evaluation_seconds=elapsed))
        print(f"Saved {path.name} in {elapsed/60:.1f} min. {len(pending)-i} remaining.", flush=True)


if __name__ == "__main__":
    main()
