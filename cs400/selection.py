"""Pure multi-objective decisions, independent of training and test data."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    candidate_id: str
    accuracy: float
    parameters: int
    seconds: float
    epoch: int

    def __post_init__(self):
        if not math.isfinite(self.accuracy) or not 0 <= self.accuracy <= 1:
            raise ValueError("Accuracy must be finite and in [0, 1]")
        if type(self.parameters) is not int or self.parameters <= 0 or type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError("Parameter count and epoch must be positive integers")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Training time must be positive and finite")

    @property
    def losses(self):
        return 1 - self.accuracy, self.parameters, self.seconds


def dominates(a, b):
    return all(x <= y for x, y in zip(a.losses, b.losses)) and any(x < y for x, y in zip(a.losses, b.losses))


def fronts(records):
    pending = list(records)
    if len({r.candidate_id for r in pending}) != len(pending):
        raise ValueError("Duplicate candidate identities")
    if len({r.epoch for r in pending}) > 1:
        raise ValueError("Only equal-budget observations are comparable")
    result = []
    while pending:
        front = [r for r in pending if not any(dominates(other, r) for other in pending)]
        result.append(sorted(front, key=lambda r: r.candidate_id))
        ids = {r.candidate_id for r in front}
        pending = [r for r in pending if r.candidate_id not in ids]
    return result


def crowding_distance(front):
    distances = {r.candidate_id: 0.0 for r in front}
    for axis in range(3):
        ordered = sorted(front, key=lambda r: (r.losses[axis], r.candidate_id))
        if not ordered:
            continue
        span = ordered[-1].losses[axis] - ordered[0].losses[axis]
        if span == 0:
            continue
        distances[ordered[0].candidate_id] = distances[ordered[-1].candidate_id] = math.inf
        for i in range(1, len(ordered) - 1):
            distances[ordered[i].candidate_id] += (ordered[i+1].losses[axis] - ordered[i-1].losses[axis]) / span
    return distances


def promote(records, slots):
    if type(slots) is not int or slots < 1:
        raise ValueError("Promotion slots must be a positive integer")
    selected = []
    for front in fronts(records):
        remaining = slots - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
        else:
            distance = crowding_distance(front)
            selected.extend(sorted(front, key=lambda r: (-distance[r.candidate_id], -r.accuracy, r.seconds, r.candidate_id))[:remaining])
    return [r.candidate_id for r in selected]


def rung_observation(candidate_id, history, parameters, full_epochs):
    if len(history) < 3 or history[-1]["epoch"] >= full_epochs:
        raise ValueError("Promotion requires at least three epochs and an intermediate rung")
    epoch = history[-1]["epoch"]
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Epoch history must be complete and consecutive")
    return Observation(candidate_id, sum(r["validation_accuracy"] for r in history[-3:]) / 3,
                       parameters, history[-1]["training_seconds"] * full_epochs / epoch, epoch)


def final_pareto(records, full_epochs):
    if any(r.epoch != full_epochs for r in records):
        raise ValueError("Partial results cannot enter final Pareto selection")
    ranked = fronts(records)
    return ranked[0] if ranked else []
