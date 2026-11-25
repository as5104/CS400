import itertools
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from cs400.comparison import compare_searches, hypervolume
from cs400.config import Protocol
from cs400.search import search_session
from cs400.storage import read_json, write_json
from test_search import FakeTrainer


class HypervolumeTests(unittest.TestCase):
    def test_boxes_duplicates_dominance_and_boundary(self):
        self.assertEqual(hypervolume([]), 0)
        self.assertAlmostEqual(hypervolume([(0.2, 0.3, 0.4)]), 0.8 * 0.7 * 0.6)
        self.assertAlmostEqual(hypervolume([(0.2, 0.3, 0.4), (0.2, 0.3, 0.4), (0.5, 0.5, 0.5)]), 0.8 * 0.7 * 0.6)
        self.assertEqual(hypervolume([(1, 0, 0)]), 0)

    def test_matches_independent_inclusion_exclusion_oracle(self):
        rng = random.Random(42)
        for _ in range(20):
            points = [tuple(rng.random() for _ in range(3)) for _ in range(5)]
            expected = 0.0
            for size in range(1, len(points) + 1):
                for subset in itertools.combinations(points, size):
                    intersection = math.prod(1 - max(p[axis] for p in subset) for axis in range(3))
                    expected += (-1) ** (size + 1) * intersection
            self.assertAlmostEqual(hypervolume(points), expected, places=12)

    def test_invalid_points_not_silently_clipped(self):
        for points in ([(0, 0, 1.1)], [(0, float('nan'), 0)], [(0, 1)]):
            with self.assertRaises(ValueError):
                hypervolume(points)


class ComparisonTests(unittest.TestCase):
    def make_search(self, directory):
        fake = FakeTrainer()
        protocol = Protocol(full_epochs=6, warmup_epochs=1, rungs=(3, 6), survivors=(1,),
                            cohort_size=2, startup_trials=2, pilot_confirmed=True, active_search_seconds=2.5)
        with patch('cs400.search.load_manifest', return_value={'id': 'fake-data'}), \
             patch('cs400.search.environment', return_value={'device': 'fake-cpu'}):
            search_session(protocol, 'unused', 'unused', directory, session_seconds=10,
                           estimated_epoch_seconds=.01, method='random', pruning=False,
                           device_name='cpu', train_fn=fake, clock=fake.clock)

    def test_frozen_archive_compares_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_search(root / 'a')
            self.make_search(root / 'b')
            report = compare_searches([root / 'a', root / 'b'], root / 'report')
            self.assertEqual(len(report['searches']), 2)
            self.assertEqual(report['searches'][0]['completed'], 2)
            self.assertGreater(report['searches'][0]['hypervolume'], 0)
            self.assertEqual(len((root / 'report/models.csv').read_text().splitlines()), 5)
            self.assertEqual(report['objective_source'], 'validation_only')

    def test_unfrozen_or_tampered_report_refused(self):
        for key, value in (('frozen', False), ('active_seconds', 500)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_search(root / 'a')
                path = root / 'a/search-results.json'
                report = read_json(path)
                report[key] = value
                write_json(path, report)
                with self.assertRaises(ValueError):
                    compare_searches([root / 'a'], root / 'report')


if __name__ == '__main__':
    unittest.main()
