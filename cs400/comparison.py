"""Compare frozen searches using validation objectives only; never train."""
import csv
from contextlib import closing
import io
import json
import math
from pathlib import Path
import sqlite3
from .config import Candidate, Protocol, digest
from .selection import Observation, final_pareto
from .storage import atomic_write, read_json, write_json


def hypervolume(points, reference=(1.0, 1.0, 1.0)):
    """Exact union volume of 3D minimization boxes [point, reference]."""
    reference = tuple(reference)
    if len(reference) != 3 or any(not math.isfinite(v) for v in reference):
        raise ValueError("Hypervolume needs a finite three-dimensional reference")
    points = [tuple(p) for p in points]
    if any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in points):
        raise ValueError("Hypervolume points must be finite three-dimensional values")
    if any(any(v > bound for v, bound in zip(p, reference)) for p in points):
        raise ValueError("A point exceeds the common hypervolume reference; use a larger reference for ALL runs")
    points = sorted(set(points))
    xs = sorted({p[0] for p in points} | {reference[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [p for p in points if p[0] <= left]
        ys = sorted({p[1] for p in active} | {reference[1]})
        area = 0.0
        for low, high in zip(ys, ys[1:]):
            z = min((p[2] for p in active if p[1] <= low), default=reference[2])
            area += (high - low) * (reference[2] - z)
        volume += (right - left) * area
    return volume


def load_search(directory):
    directory = Path(directory).resolve()
    report = read_json(directory / "search-results.json")
    if not report["frozen"]:
        raise ValueError("Only frozen searches qualify for a final comparison")
    with closing(sqlite3.connect((directory / "search.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        state = json.loads(db.execute("SELECT value FROM metadata WHERE key='controller'").fetchone()[0])
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        stored = {cid: json.loads(obs) for cid, obs in db.execute("SELECT id, observation FROM completed")}
        candidates = list(db.execute("SELECT id,cohort FROM candidates"))
        completions = sorted(db.execute("SELECT id,active_seconds FROM completed"), key=lambda r: (r[1], r[0]))
    if state["in_session"] or not state["finished"] or state.get("requires_reconciliation"):
        raise ValueError("Search has unfinished or uncertain accounting")
    if state["active_seconds"] != report["active_seconds"]:
        raise ValueError("Search report and ledger active times disagree")
    if (identity["protocol"] != report["protocol"] or identity["manifest_id"] != report["manifest_id"]
            or identity["hardware_id"] != digest(report["hardware"])):
        raise ValueError("Search metadata does not match its ledger")
    protocol = Protocol(**report["protocol"])
    observations = []
    for row in report["completed"]:
        obs = Observation(**row["metrics"])
        if Candidate(**row["candidate"]).id != obs.candidate_id or obs.epoch != protocol.full_epochs:
            raise ValueError("Completed archive contains an invalid configuration or partial result")
        if stored.get(obs.candidate_id) != row["metrics"]:
            raise ValueError("Completed metrics differ from the ledger")
        observations.append(obs)
    if len(observations) != len(stored) or len({o.candidate_id for o in observations}) != len(observations):
        raise ValueError("Completed archive has missing or duplicate models")
    if {o.candidate_id for o in final_pareto(observations, protocol.full_epochs)} != {o["candidate_id"] for o in report["pareto"]}:
        raise ValueError("Saved Pareto set disagrees with completed objectives")
    if any(not math.isfinite(t) or t < 0 or t > protocol.active_search_seconds for _, t in completions):
        raise ValueError("An archived model completed outside the active-time budget")
    return directory, report, state["settings"], observations, candidates, completions


def compare_searches(directories, output_dir, parameter_reference=6_000_000, seconds_reference=3600.0):
    if any(not math.isfinite(v) or v <= 0 for v in (parameter_reference, seconds_reference)):
        raise ValueError("Reference scales must be positive and finite")
    loaded = [load_search(p) for p in directories]
    if not loaded or len({row[0] for row in loaded}) != len(loaded):
        raise ValueError("Supply at least one distinct search directory")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("Comparison output exists; choose a new directory")
    base = None
    summaries, models = [], []
    def point(o):
        return (1 - o.accuracy, o.parameters / parameter_reference, o.seconds / seconds_reference)
    for directory, report, settings, observations, candidates, completions in loaded:
        protocol = {k: v for k, v in report["protocol"].items() if k not in ("seed", "pilot_confirmed")}
        signature = dict(protocol=protocol, hardware=report["hardware"], manifest_id=report["manifest_id"],
                         epoch_cost_seconds=settings["estimated_epoch_seconds"])
        if base is None:
            base = signature
        elif signature != base:
            raise ValueError("Comparisons require matching protocol, time cap, data, hardware/code and admission estimate; only seed/method/pruning may differ")
        by_id = {o.candidate_id: o for o in observations}
        progress, completed_so_far = [], []
        for cid, seconds in completions:
            completed_so_far.append(point(by_id[cid]))
            progress.append(dict(active_seconds=seconds, hypervolume=hypervolume(completed_so_far)))
        pareto = final_pareto(observations, report["protocol"]["full_epochs"])
        pareto_ids = {o.candidate_id for o in pareto}
        budget = report["protocol"]["active_search_seconds"]
        summaries.append(dict(run=str(directory), method=settings["method"], pruning=settings["pruning"],
                              seed=report["protocol"]["seed"], active_budget_seconds=budget,
                              active_seconds=report["active_seconds"],
                              budget_utilization=min(report["active_seconds"], budget) / budget,
                              overrun_seconds=max(0.0, report["active_seconds"] - budget),
                              stop_reason=report["stop_reason"], proposed_candidates=len(candidates),
                              completed=len(observations), pareto_count=len(pareto),
                              best_validation_accuracy=max((o.accuracy for o in observations), default=None),
                              hypervolume=hypervolume([point(o) for o in observations]),
                              hypervolume_history=progress))
        for row in report["completed"]:
            o = row["metrics"]
            models.append(dict(run=str(directory), method=settings["method"], pruning=settings["pruning"],
                               seed=report["protocol"]["seed"], **row["candidate"],
                               candidate_id=o["candidate_id"], validation_accuracy=o["accuracy"],
                               parameters=o["parameters"], training_seconds=o["seconds"],
                               epoch=o["epoch"], pareto=o["candidate_id"] in pareto_ids))
    result = dict(schema=1, objective_source="validation_only", common_settings=base,
                  hypervolume_definition="Union of minimization boxes in (1-accuracy, parameters/parameter_reference, training_seconds/seconds_reference)",
                  reference=dict(error=1.0, parameter_reference=parameter_reference, seconds_reference=seconds_reference),
                  interpretation="Higher hypervolume is better at this common reference; single-seed results are descriptive, not evidence of statistical superiority. Report unused budgets alongside final scores.",
                  searches=summaries)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "comparison.json", result)
    stream = io.StringIO(newline="")
    if models:
        writer = csv.DictWriter(stream, fieldnames=list(models[0]))
        writer.writeheader()
        writer.writerows(models)
    atomic_write(target / "models.csv", lambda file: file.write(stream.getvalue().encode("utf-8")))
    return result
