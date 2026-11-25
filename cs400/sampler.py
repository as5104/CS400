"""Completed-only MOTPE proposal history; no Optuna multi-objective pruner."""
from dataclasses import asdict
import math
import random
from .config import Candidate, DEPTH_NAMES
from .selection import Observation


def random_candidate(seed, optimizer=None):
    rng = random.Random(seed)
    optimizer = optimizer or rng.choice(("sgd", "adamw"))
    lo, hi = (0.01, 0.1) if optimizer == "sgd" else (0.0001, 0.003)
    return Candidate(rng.choice(DEPTH_NAMES), rng.choice((16, 24, 32)), optimizer,
                     math.exp(rng.uniform(math.log(lo), math.log(hi))),
                     math.exp(rng.uniform(math.log(1e-6), math.log(0.01))),
                     rng.choice((0.0, 0.1, 0.2)), rng.choice(("basic", "erasing")))


def startup_candidates(count, seed):
    if count < 2:
        raise ValueError("Startup needs at least two candidates to cover both optimizers")
    configs = [Candidate()]
    # Reference is AdamW; alternate to give both families balanced coverage.
    configs.extend(random_candidate(seed + i, "sgd" if i % 2 else "adamw") for i in range(1, count))
    return configs


def distributions(candidate):
    from optuna.distributions import CategoricalDistribution as C, FloatDistribution as F
    lo, hi = (0.01, 0.1) if candidate.optimizer == "sgd" else (0.0001, 0.003)
    return dict(depth=C(DEPTH_NAMES), width=C((16, 24, 32)), optimizer=C(("sgd", "adamw")),
                **{f"lr_{candidate.optimizer}": F(lo, hi, log=True)},
                weight_decay=F(1e-6, 0.01, log=True), dropout=C((0.0, 0.1, 0.2)), augmentation=C(("basic", "erasing")))


def parameters(candidate):
    result = asdict(candidate)
    result[f"lr_{candidate.optimizer}"] = result.pop("learning_rate")
    return result


def propose(completed, *, full_epochs, startup_trials, seed, excluded_ids=()):
    import optuna
    if optuna.__version__ != "4.5.0":
        raise RuntimeError("This protocol is verified against Optuna 4.5.0; pin that version")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for candidate, observation in completed:
        if observation.epoch != full_epochs or candidate.id != observation.candidate_id:
            raise ValueError("MOTPE history accepts only matching full-budget observations")
    excluded = set(excluded_ids) | {c.id for c, _ in completed}
    for attempt in range(21):
        sampler = optuna.samplers.TPESampler(seed=seed + attempt, n_startup_trials=startup_trials,
                                             n_ei_candidates=24, multivariate=True, group=True, constant_liar=False)
        study = optuna.create_study(directions=["maximize", "minimize", "minimize"], sampler=sampler)
        for candidate, observation in completed:
            study.add_trial(optuna.trial.create_trial(params=parameters(candidate), distributions=distributions(candidate),
                            values=[observation.accuracy, observation.parameters, observation.seconds]))
        trial = study.ask()
        depth = trial.suggest_categorical("depth", DEPTH_NAMES)
        width = trial.suggest_categorical("width", (16, 24, 32))
        optimizer = trial.suggest_categorical("optimizer", ("sgd", "adamw"))
        lo, hi = (0.01, 0.1) if optimizer == "sgd" else (0.0001, 0.003)
        config = Candidate(depth, width, optimizer, trial.suggest_float(f"lr_{optimizer}", lo, hi, log=True),
                           trial.suggest_float("weight_decay", 1e-6, 0.01, log=True),
                           trial.suggest_categorical("dropout", (0.0, 0.1, 0.2)),
                           trial.suggest_categorical("augmentation", ("basic", "erasing")))
        if config.id not in excluded:
            return config
    for attempt in range(100):
        config = random_candidate(seed + 1000 + attempt)
        if config.id not in excluded:
            return config
    raise RuntimeError("Could not propose a distinct configuration")

