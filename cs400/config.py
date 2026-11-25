"""Immutable experiment settings and explicit initial search space."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DEPTHS = ((1, 1, 1, 1), (1, 2, 2, 1), (2, 2, 2, 2), (2, 3, 3, 2), (3, 3, 3, 3))
DEPTH_NAMES = tuple("-".join(map(str, depth)) for depth in DEPTHS)
SCHEMA_VERSION = 1


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    depth: str = "2-2-2-2"
    width: int = 32
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    dropout: float = 0.0
    augmentation: str = "basic"

    def __post_init__(self):
        if self.depth not in DEPTH_NAMES or self.width not in (16, 24, 32):
            raise ValueError("Architecture is outside the approved initial space")
        if self.optimizer not in ("sgd", "adamw"):
            raise ValueError("Optimizer must be sgd or adamw")
        lo, hi = (0.01, 0.1) if self.optimizer == "sgd" else (0.0001, 0.003)
        if not math.isfinite(self.learning_rate) or not lo <= self.learning_rate <= hi:
            raise ValueError("Learning rate is outside the optimizer's range")
        if not math.isfinite(self.weight_decay) or not 1e-6 <= self.weight_decay <= 0.01:
            raise ValueError("Weight decay must be in [1e-6, 0.01]")
        if self.dropout not in (0.0, 0.1, 0.2):
            raise ValueError("Dropout must be 0, 0.1 or 0.2")
        if self.augmentation not in ("basic", "erasing"):
            raise ValueError("Augmentation must be basic or erasing")

    @property
    def id(self):
        return digest(asdict(self))

    @property
    def blocks(self):
        return tuple(map(int, self.depth.split("-")))


@dataclass(frozen=True)
class Protocol:
    full_epochs: int = 60
    warmup_epochs: int = 5
    batch_size: int = 128
    precision: str = "fp32"
    workers: int = 0
    seed: int = 42
    rungs: tuple = (15, 30, 60)
    survivors: tuple = (6, 3)
    cohort_size: int = 12
    startup_trials: int = 12
    max_parameters: int = 6_000_000
    deterministic: bool = True
    # Pilot numbers deliberately do not authorize a main search.
    pilot_confirmed: bool = False
    active_search_seconds: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "rungs", tuple(self.rungs))
        object.__setattr__(self, "survivors", tuple(self.survivors))
        if type(self.deterministic) is not bool or type(self.pilot_confirmed) is not bool:
            raise ValueError("deterministic and pilot_confirmed must be JSON booleans")
        for name in ("full_epochs", "warmup_epochs", "batch_size", "cohort_size", "startup_trials", "max_parameters"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.workers) is not int or self.workers < 0 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("workers and seed must be nonnegative integers")
        if not self.warmup_epochs < self.full_epochs:
            raise ValueError("Warmup must be shorter than the full training budget")
        if self.precision not in ("fp32", "amp"):
            raise ValueError("precision must be fp32 or amp")
        if not self.rungs or any(type(r) is not int or r < 3 for r in self.rungs):
            raise ValueError("Rungs need integer epoch budgets of at least 3")
        if tuple(sorted(set(self.rungs))) != self.rungs or self.rungs[-1] != self.full_epochs:
            raise ValueError("Rungs must increase and end at the full budget")
        if len(self.survivors) != len(self.rungs) - 1:
            raise ValueError("Each intermediate rung needs one survivor count")
        previous = self.cohort_size
        for n in self.survivors:
            if type(n) is not int or not 0 < n < previous:
                raise ValueError("Survivor counts must strictly decrease")
            previous = n
        if self.active_search_seconds is not None and (
            not math.isfinite(self.active_search_seconds) or self.active_search_seconds <= 0
        ):
            raise ValueError("Active search budget must be positive and finite")

    @property
    def id(self):
        return digest(asdict(self))

    def learning_rate(self, base, epoch):
        if not 1 <= epoch <= self.full_epochs:
            raise ValueError("Epoch outside the full-budget schedule")
        if epoch <= self.warmup_epochs:
            return base * epoch / self.warmup_epochs
        fraction = (epoch - self.warmup_epochs) / (self.full_epochs - self.warmup_epochs)
        return base * (0.01 + 0.99 * (1 + math.cos(math.pi * fraction)) / 2)

