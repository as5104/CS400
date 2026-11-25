"""Manual synchronous search sessions with a persisted cohort and time budget."""
from dataclasses import asdict
import math
from pathlib import Path
import time
import torch
from .config import digest
from .controller import Cohort
from .data import load_manifest
from .ledger import Ledger
from .sampler import propose, random_candidate, startup_candidates
from .selection import Observation, rung_observation
from .storage import read_json, run_lock, write_json
from .trainer import environment, train_candidate


def cohort_work(protocol, pruning):
    if not pruning:
        return protocol.cohort_size * protocol.full_epochs
    previous = cost = 0
    for epoch, count in zip(protocol.rungs, (protocol.cohort_size, *protocol.survivors)):
        cost += (epoch - previous) * count
        previous = epoch
    return cost


def search_session(protocol, data_root, manifest_path, directory, *, session_seconds,
                   estimated_epoch_seconds, method="motpe", pruning=True, device_name="cuda",
                   resume=False, train_fn=train_candidate, clock=time.perf_counter, pause_check=None):
    if not protocol.pilot_confirmed or protocol.active_search_seconds is None:
        raise ValueError("Run the manual pilot first; freeze its settings and choose an active search budget")
    if method not in ("motpe", "random"):
        raise ValueError("Method must be motpe or random")
    if any(not math.isfinite(v) or v <= 0 for v in (session_seconds, estimated_epoch_seconds)):
        raise ValueError("Session time and calibrated epoch-cost estimate must be positive and finite")
    manifest = load_manifest(manifest_path)
    hardware = environment(torch.device(device_name))
    directory = Path(directory)
    with run_lock(directory):
        ledger = Ledger(directory / "search.sqlite", protocol, manifest["id"], digest(hardware))
        try:
            state = ledger.state()
            settings = dict(method=method, pruning=pruning, estimated_epoch_seconds=estimated_epoch_seconds)
            if state:
                if not resume:
                    raise FileExistsError("Search already exists; explicitly use --resume")
                if state["settings"] != settings:
                    raise ValueError("Search method or calibrated cost estimate changed")
                if state["in_session"] or state.get("requires_reconciliation"):
                    raise RuntimeError("Unexpected shutdown left uncertain active-time accounting. Do not resume this as a controlled experiment; candidate checkpoints remain available for recovery/inspection")
                if state["finished"]:
                    return read_json(directory / "search-results.json")
            elif resume:
                raise FileNotFoundError("No saved search state to resume")
            else:
                minimum_cost = (protocol.startup_trials * protocol.full_epochs + cohort_work(protocol, pruning)) * estimated_epoch_seconds
                if minimum_cost > protocol.active_search_seconds:
                    raise ValueError("Budget estimate cannot cover startup and one subsequent cohort; revise the calibrated protocol")
                state = dict(settings=settings, active_seconds=0.0, in_session=False, finished=False,
                             cohort_number=-1, cohort=None, proposal_number=0, seen_ids=[], stop_reason=None)
                ledger.save_state(state)
            base_active = state["active_seconds"]
            started = clock()
            def elapsed():
                return max(0.0, clock() - started)
            def persist():
                state["active_seconds"] = base_active + elapsed()
                ledger.save_state(state)
            state["in_session"] = True
            persist()
            try:
                while True:
                    if pause_check is not None and pause_check():
                        state["stop_reason"] = "session_or_requested_pause"
                        break
                    active = base_active + elapsed()
                    if active >= protocol.active_search_seconds:
                        state.update(finished=True, stop_reason="active_budget_reached")
                        break
                    if elapsed() >= session_seconds:
                        state["stop_reason"] = "session_limit"
                        break
                    if state["cohort"] is None:
                        number = state["cohort_number"] + 1
                        is_startup = number == 0
                        work = protocol.startup_trials * protocol.full_epochs if is_startup else cohort_work(protocol, pruning)
                        if work * estimated_epoch_seconds > protocol.active_search_seconds - active:
                            state.update(finished=True, stop_reason="insufficient_estimated_budget_for_next_cohort")
                            break
                        if is_startup:
                            candidates = startup_candidates(protocol.startup_trials, protocol.seed)
                        else:
                            candidates = []
                            archive = ledger.archive()
                            excluded = set(state["seen_ids"])
                            for _ in range(protocol.cohort_size):
                                sequence = state["proposal_number"]
                                seed = (protocol.seed + 1009 * (sequence + 1)) % (2**32)
                                if method == "motpe":
                                    candidate = propose(archive, full_epochs=protocol.full_epochs,
                                                        startup_trials=protocol.startup_trials, seed=seed, excluded_ids=excluded)
                                else:
                                    candidate = random_candidate(seed)
                                    retry = 0
                                    while candidate.id in excluded:
                                        retry += 1
                                        candidate = random_candidate(seed + retry)
                                state["proposal_number"] += 1
                                excluded.add(candidate.id)
                                candidates.append(candidate)
                        # Deterministic order reduces systematic first/last timing effects.
                        import random
                        random.Random(protocol.seed + number).shuffle(candidates)
                        cohort = Cohort(protocol, candidates, startup=is_startup, full_training=not pruning)
                        ledger.add_cohort(number, candidates)
                        state.update(cohort_number=number, cohort=cohort.state())
                        state["seen_ids"].extend(c.id for c in candidates)
                        persist()
                    cohort = Cohort.restore(state["cohort"])
                    jobs = cohort.jobs()
                    if not jobs:
                        selected = cohort.advance()
                        if cohort.finished and cohort.startup and len(selected) < protocol.startup_trials:
                            state.update(cohort=cohort.state(), finished=True, stop_reason="startup_incomplete_inspect_failures")
                            persist()
                            break
                        if cohort.decisions and not cohort.finished:
                            ledger.record_promotion(state["cohort_number"], cohort.decisions[-1]["epoch"], selected)
                        state["cohort"] = None if cohort.finished else cohort.state()
                        persist()
                        continue
                    # At each new rung, check the cost of extending all survivors.
                    remaining_work = sum(target - (cohort.history.get(c.id, [{}])[-1].get("epoch", 0)) for c, target in jobs)
                    if remaining_work * estimated_epoch_seconds > protocol.active_search_seconds - (base_active + elapsed()):
                        state.update(finished=True, stop_reason="insufficient_estimated_budget_for_rung")
                        break
                    candidate, target = jobs[0]
                    trial_dir = directory / "trials" / candidate.id
                    remaining_seconds = min(session_seconds - elapsed(), protocol.active_search_seconds - base_active - elapsed())
                    if remaining_seconds <= 0:
                        continue
                    try:
                        result = train_fn(candidate, protocol, data_root, manifest_path, trial_dir,
                                          target_epoch=target, session_seconds=remaining_seconds, device_name=device_name,
                                          resume=(trial_dir / "run.json").exists(),
                                          **({"pause_check": pause_check} if pause_check is not None else {}))
                    except (torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
                        cohort.fail(candidate.id, f"{type(exc).__name__}: {exc}")
                        state["cohort"] = cohort.state()
                        persist()
                        if device_name == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    if result.get("accounting_uncertain"):
                        raise RuntimeError("Candidate has uncertain interruption cost; cannot include it in controlled search")
                    if result["epoch"]:
                        history = read_json(trial_dir / "history.json")
                        cohort.record(candidate.id, history, result["parameters"])
                        if result["epoch"] == target:
                            obs = (Observation(candidate.id, result["validation_accuracy"], result["parameters"], result["training_seconds"], target)
                                   if target == protocol.full_epochs else rung_observation(candidate.id, history, result["parameters"], protocol.full_epochs))
                            ledger.record_rung(obs)
                            completion_time = base_active + elapsed()
                            if target == protocol.full_epochs and completion_time <= protocol.active_search_seconds:
                                ledger.complete(obs, completion_time)
                    state["cohort"] = cohort.state()
                    persist()
                    if result["status"] == "paused" or result.get("pause_requested"):
                        state["stop_reason"] = "session_or_requested_pause"
                        break
            except BaseException:
                state["requires_reconciliation"] = True
                raise
            finally:
                state["in_session"] = False
                persist()
            archive = ledger.archive()
            report = dict(schema=1, frozen=state["finished"], stop_reason=state["stop_reason"],
                          protocol=asdict(protocol), manifest_id=manifest["id"], hardware=hardware,
                          active_seconds=state["active_seconds"],
                          overrun_seconds=max(0.0, state["active_seconds"] - protocol.active_search_seconds),
                          completed=[dict(candidate=asdict(c), metrics=asdict(o), run_dir=str((directory / "trials" / c.id).resolve())) for c,o in archive],
                          pareto=[asdict(o) for o in ledger.pareto()])
            write_json(directory / "search-results.json", report)
            return report
        finally:
            ledger.close()
