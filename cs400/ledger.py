"""Transactional search records with a separate eligible full-budget archive."""
from dataclasses import asdict
import json
import sqlite3
from .config import Candidate, Protocol
from .selection import Observation, final_pareto


class Ledger:
    def __init__(self, path, protocol: Protocol, manifest_id, hardware_id):
        self.db = sqlite3.connect(path)
        self.protocol = protocol
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, config TEXT NOT NULL, cohort INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS rungs (id TEXT NOT NULL, epoch INTEGER NOT NULL, observation TEXT NOT NULL,
                                            PRIMARY KEY(id, epoch));
            CREATE TABLE IF NOT EXISTS completed (id TEXT PRIMARY KEY, observation TEXT NOT NULL, active_seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS decisions (cohort INTEGER, epoch INTEGER, selected TEXT, PRIMARY KEY(cohort, epoch));
        ''')
        identity = json.dumps(dict(protocol=asdict(protocol), manifest_id=manifest_id, hardware_id=hardware_id), sort_keys=True)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if previous and previous[0] != identity:
            self.db.close()
            raise ValueError("Cannot mix protocols, datasets, or hardware in one ledger")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (identity,))

    def close(self):
        self.db.close()

    def state(self):
        row = self.db.execute("SELECT value FROM metadata WHERE key='controller'").fetchone()
        return json.loads(row[0]) if row else None

    def save_state(self, state):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('controller',?)", (json.dumps(state, allow_nan=False),))

    def add_cohort(self, number, candidates):
        with self.db:
            for candidate in candidates:
                self.db.execute("INSERT INTO candidates VALUES (?,?,?)", (candidate.id, json.dumps(asdict(candidate)), number))

    def record_rung(self, observation):
        if observation.epoch not in self.protocol.rungs:
            raise ValueError("Observation is not at a declared rung")
        if not self.db.execute("SELECT 1 FROM candidates WHERE id=?", (observation.candidate_id,)).fetchone():
            raise ValueError("Unknown candidate")
        with self.db:
            self.db.execute("INSERT INTO rungs VALUES (?,?,?)", (observation.candidate_id, observation.epoch, json.dumps(asdict(observation))))

    def record_promotion(self, cohort, epoch, selected):
        eligible = {row[0] for row in self.db.execute(
            "SELECT c.id FROM candidates c JOIN rungs r ON c.id=r.id WHERE c.cohort=? AND r.epoch=?", (cohort, epoch))}
        if epoch not in self.protocol.rungs[:-1] or len(set(selected)) != len(selected) or not set(selected) <= eligible:
            raise ValueError("Promotion IDs must be unique valid members at an intermediate rung")
        if len(selected) > self.protocol.survivors[self.protocol.rungs.index(epoch)]:
            raise ValueError("Promotion exceeds its quota")
        with self.db:
            self.db.execute("INSERT INTO decisions VALUES (?,?,?)", (cohort, epoch, json.dumps(selected)))

    def complete(self, observation, active_seconds):
        import math
        if observation.epoch != self.protocol.full_epochs or not math.isfinite(active_seconds) or active_seconds < 0:
            raise ValueError("Completion requires real full-budget metrics and finite active cost")
        if self.protocol.active_search_seconds is not None and active_seconds > self.protocol.active_search_seconds:
            raise ValueError("Result completed after the search cutoff")
        if not self.db.execute("SELECT 1 FROM candidates WHERE id=?", (observation.candidate_id,)).fetchone():
            raise ValueError("Unknown candidate")
        with self.db:
            self.db.execute("INSERT INTO completed VALUES (?,?,?)", (observation.candidate_id, json.dumps(asdict(observation)), active_seconds))

    def archive(self):
        return [(Candidate(**json.loads(config)), Observation(**json.loads(obs))) for config, obs in
                self.db.execute("SELECT c.config, r.observation FROM candidates c JOIN completed r ON c.id=r.id ORDER BY r.active_seconds,c.id")]

    def pareto(self):
        return final_pareto([o for _, o in self.archive()], self.protocol.full_epochs)
