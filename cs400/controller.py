"""Synchronous cohort state machine; the trainer is driven by explicit callers.

No training occurs here. A durable caller saves state before dispatching each
job; replaying a finished job with different measurements is rejected.
"""
from dataclasses import asdict, dataclass, field
from .config import Candidate, Protocol
from .selection import Observation, promote, rung_observation


@dataclass
class Cohort:
    protocol: Protocol
    candidates: list[Candidate]
    startup: bool = False
    full_training: bool = False
    rung_index: int = 0
    active: list[str] = field(default_factory=list)
    history: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    finished: bool = False

    def __post_init__(self):
        ids = [c.id for c in self.candidates]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Cohort requires distinct candidates")
        expected = self.protocol.startup_trials if self.startup else self.protocol.cohort_size
        if len(ids) != expected:
            raise ValueError("Cohort size differs from the protocol")
        if not self.active and not self.finished:
            self.active = ids

    @property
    def target_epoch(self):
        return self.protocol.full_epochs if self.startup or self.full_training else self.protocol.rungs[self.rung_index]

    def jobs(self):
        if self.finished:
            return []
        configs = {c.id: c for c in self.candidates}
        return [(configs[cid], self.target_epoch) for cid in self.active
                if cid not in self.failures and (not self.history.get(cid) or self.history[cid][-1]["epoch"] < self.target_epoch)]

    def record(self, candidate_id, history, parameters):
        if self.finished or candidate_id not in self.active or candidate_id in self.failures:
            raise ValueError("Candidate is not active")
        if not history or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
            raise ValueError("History must contain consecutive completed epochs")
        if history[-1]["epoch"] > self.target_epoch:
            raise ValueError("Candidate passed its assigned rung")
        old = self.history.get(candidate_id, [])
        if len(history) < len(old) or history[:len(old)] != old:
            raise ValueError("Previously committed epochs cannot change")
        prior_time = 0.0
        for row in history:
            Observation(candidate_id, row["validation_accuracy"], parameters, row["training_seconds"], row["epoch"])
            if row["training_seconds"] <= prior_time:
                raise ValueError("Accumulated training time must increase")
            prior_time = row["training_seconds"]
        if candidate_id in self.parameters and self.parameters[candidate_id] != parameters:
            raise ValueError("Model size changed during training")
        self.history[candidate_id] = [dict(r) for r in history]
        self.parameters[candidate_id] = parameters

    def fail(self, candidate_id, reason):
        if self.finished or candidate_id not in self.active:
            raise ValueError("Candidate is not active")
        self.failures[candidate_id] = str(reason)

    def advance(self):
        if self.finished or self.jobs():
            raise ValueError("Wait until every valid candidate reaches the current rung")
        eligible = [cid for cid in self.active if cid not in self.failures]
        if self.target_epoch == self.protocol.full_epochs or not eligible:
            self.active = eligible
            self.finished = True
            return eligible
        observations = [rung_observation(cid, self.history[cid], self.parameters[cid], self.protocol.full_epochs) for cid in eligible]
        slots = self.protocol.survivors[self.rung_index]
        selected = promote(observations, slots)
        self.decisions.append(dict(epoch=self.target_epoch, selected=selected,
                                   stopped=[cid for cid in eligible if cid not in selected],
                                   observations=[asdict(o) for o in observations]))
        self.active = selected
        self.rung_index += 1
        return selected

    def completed(self):
        if not self.finished:
            raise ValueError("Cohort has not finished")
        return [Observation(cid, self.history[cid][-1]["validation_accuracy"], self.parameters[cid],
                            self.history[cid][-1]["training_seconds"], self.protocol.full_epochs)
                for cid in self.active]

    def state(self):
        return asdict(self)

    @classmethod
    def restore(cls, state):
        copy = dict(state)
        copy["protocol"] = Protocol(**copy["protocol"])
        copy["candidates"] = [Candidate(**c) for c in copy["candidates"]]
        return cls(**copy)

