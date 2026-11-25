"""Search integration using a fake clock/trainer; never performs learning."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cs400.config import Protocol
from cs400.search import search_session
from cs400.storage import read_json, write_json


class FakeTrainer:
    def __init__(self):
        self.now = 0.0
        self.calls = []
        self.pause = False

    def clock(self):
        return self.now

    def __call__(self, candidate, protocol, data_root, manifest, directory, *, target_epoch, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        history_path = directory / 'history.json'
        history = read_json(history_path) if history_path.exists() else []
        self.calls.append((candidate.id,len(history),target_epoch))
        for epoch in range(len(history)+1,target_epoch+1):
            history.append(dict(epoch=epoch,validation_accuracy=.6,training_seconds=float(epoch)))
        write_json(directory/'run.json',{})
        write_json(history_path,history)
        self.now += 1.0
        return dict(epoch=target_epoch,parameters=100,validation_accuracy=.6,training_seconds=float(target_epoch),
                    status='complete' if target_epoch==protocol.full_epochs else 'target_reached',
                    accounting_uncertain=False,pause_requested=self.pause)


class SearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.protocol=Protocol(full_epochs=6,warmup_epochs=1,rungs=(3,6),survivors=(1,),cohort_size=2,startup_trials=2,
                               pilot_confirmed=True,active_search_seconds=30)
        self.fake=FakeTrainer()

    def run_session(self,path,**overrides):
        options=dict(session_seconds=3,estimated_epoch_seconds=.1,method='random',device_name='cpu',
                     train_fn=self.fake,clock=self.fake.clock)
        options.update(overrides)
        with patch('cs400.search.load_manifest',return_value={'id':'fake-data'}), \
             patch('cs400.search.environment',return_value={'device':'fake-cpu'}):
            return search_session(self.protocol,'unused','unused',path,**options)

    def test_sessions_resume_same_cohort_without_retraining_startup(self):
        with tempfile.TemporaryDirectory() as temp:
            first=self.run_session(temp)
            self.assertEqual(len(first['completed']),2)
            self.assertFalse(first['frozen'])
            self.assertEqual(first['active_seconds'],3)
            before=list(self.fake.calls)
            second=self.run_session(temp,resume=True)
            self.assertEqual(second['active_seconds'],6)
            self.assertGreaterEqual(len(second['completed']),3)
            self.assertTrue(all(c not in self.fake.calls[len(before):] for c in before[:2]))
            self.assertTrue(any(start==3 and target==6 for _,start,target in self.fake.calls))

    def test_cutoff_excludes_late_completion(self):
        self.protocol=replace(self.protocol,active_search_seconds=2.5)
        with tempfile.TemporaryDirectory() as temp:
            result=self.run_session(temp,session_seconds=10,estimated_epoch_seconds=.01,pruning=False)
            self.assertTrue(result['frozen'])
            self.assertEqual(len(result['completed']),2)
            self.assertGreater(result['overrun_seconds'],0)

    def test_ctrl_c_at_rung_end_still_stops_session(self):
        self.fake.pause=True
        with tempfile.TemporaryDirectory() as temp:
            result=self.run_session(temp)
            self.assertEqual(len(self.fake.calls),1)
            self.assertEqual(result['stop_reason'],'session_or_requested_pause')

    def test_external_pause_preserves_resumable_search(self):
        with tempfile.TemporaryDirectory() as temp:
            result=self.run_session(temp,pause_check=lambda:len(self.fake.calls)>=1)
            self.assertEqual(len(self.fake.calls),1)
            self.assertEqual(result['stop_reason'],'session_or_requested_pause')
            resumed=self.run_session(temp,resume=True,pause_check=lambda:False)
            self.assertGreater(len(resumed['completed']),len(result['completed']))

    def test_unconfirmed_protocol_cannot_train(self):
        self.protocol=replace(self.protocol,pilot_confirmed=False)
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):self.run_session(temp)
        self.assertFalse(self.fake.calls)

    def test_changed_settings_and_missing_resume_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_session(temp)
            with self.assertRaises(FileExistsError):self.run_session(temp)
            with self.assertRaises(ValueError):self.run_session(temp,resume=True,estimated_epoch_seconds=.2)

    def test_full_budget_baseline_has_no_intermediate_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_session(temp,pruning=False,session_seconds=5)
            self.assertEqual({target for _,_,target in self.fake.calls},{6})


if __name__=='__main__':
    unittest.main()
