"""Non-training tests: pure decisions, file recovery and CPU forward inference."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import torch

from cs400.config import Candidate, Protocol, DEPTH_NAMES, digest
from cs400.controller import Cohort
from cs400.data import stratified_indices, validate_indices
from cs400.evaluation import CORRUPTIONS, robustness_metrics
from cs400.ledger import Ledger
from cs400.model import ResNet, make_optimizer, parameter_count
from cs400.sampler import propose, startup_candidates
from cs400.selection import Observation, crowding_distance, final_pareto, fronts, promote, rung_observation
from cs400.storage import atomic_write, read_json, write_json, run_lock
from cs400.trainer import load_checkpoint, random_states, restore_random_states, save_checkpoint, train_candidate

torch.set_num_threads(1)


class ConfigTests(unittest.TestCase):
    def test_candidate_identity_roundtrip(self):
        value = Candidate()
        self.assertEqual(value, Candidate(**asdict(value)))
        self.assertNotEqual(value.id, replace(value, width=16).id)

    def test_invalid_candidate_values(self):
        for kwargs in (dict(width=64), dict(learning_rate=float('nan')), dict(optimizer='adam'), dict(dropout=0.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Candidate(**kwargs)

    def test_protocol_and_schedule_boundaries(self):
        p = Protocol()
        self.assertAlmostEqual(p.learning_rate(.1, 1), .02)
        self.assertAlmostEqual(p.learning_rate(.1, 5), .1)
        self.assertAlmostEqual(p.learning_rate(.1, 60), .001)
        self.assertEqual(p.id, Protocol(**json.loads(json.dumps(asdict(p)))).id)
        for kwargs in (dict(rungs=(15, 15, 60)), dict(survivors=(6, 7)), dict(precision='auto'), dict(active_search_seconds=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Protocol(**kwargs)


class SplitTests(unittest.TestCase):
    def test_cifar_split_no_overlap_and_exact_class_balance(self):
        labels = np.repeat(np.arange(10), 5000)
        train, val = stratified_indices(labels, 500, 42)
        self.assertEqual((len(train), len(val)), (45000, 5000))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(np.bincount(labels[val]).tolist(), [500] * 10)
        self.assertEqual((train, val), stratified_indices(labels, 500, 42))
        self.assertNotEqual(val, stratified_indices(labels, 500, 43)[1])

    def test_reject_overlap_and_missing_indices(self):
        for train, val in (([0, 1], [1, 2]), ([0], [2]), ([0, 1], [3])):
            with self.assertRaises(ValueError):
                validate_indices(train, val, 3)


class ModelTests(unittest.TestCase):
    def test_all_architectures_fit_parameter_cap(self):
        for depth in DEPTH_NAMES:
            for width in (16, 24, 32):
                with self.subTest(depth=depth, width=width):
                    model = ResNet(Candidate(depth=depth, width=width))
                    self.assertLess(parameter_count(model), 6_000_000)
                    self.assertEqual(model.stem[0].kernel_size, (3, 3))

    def test_cpu_forward_shape_without_training(self):
        model = ResNet(Candidate(width=16), classes=7).eval()
        with torch.inference_mode():
            scores = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(scores.shape), (2, 7))
        self.assertTrue(torch.isfinite(scores).all())

    def test_bias_and_normalization_excluded_from_decay(self):
        model = ResNet(Candidate(width=16))
        for optimizer_name in ('sgd', 'adamw'):
            lr = .03 if optimizer_name == 'sgd' else .001
            optimizer = make_optimizer(model, Candidate(width=16, optimizer=optimizer_name, learning_rate=lr))
            for group in optimizer.param_groups:
                if group['weight_decay']:
                    self.assertTrue(all(p.ndim > 1 for p in group['params']))
                else:
                    self.assertTrue(all(p.ndim <= 1 for p in group['params']))


class SelectionTests(unittest.TestCase):
    def obs(self, name, a, p, t, e=15):
        return Observation(name, a, p, t, e)

    def test_dominance_and_multiple_fronts(self):
        a, b, c = self.obs('a', .9, 100, 10), self.obs('b', .8, 200, 20), self.obs('c', .7, 50, 5)
        self.assertEqual([{r.candidate_id for r in f} for f in fronts([b,c,a])], [{'a','c'}, {'b'}])
        self.assertEqual(set(promote([b,c,a], 2)), {'a','c'})

    def test_many_nondominated_candidates_exact_quota(self):
        points = [self.obs(str(i), .5+i*.02, 100+i*10, 10+i) for i in range(12)]
        selected = promote(points, 6)
        self.assertEqual(len(selected), 6)
        self.assertTrue({'0','11'} <= set(selected))
        self.assertEqual(selected, promote(list(reversed(points)), 6))

    def test_constant_axes_and_ties(self):
        points = [self.obs(x, .8, 100, 10) for x in ('c','b','a')]
        self.assertEqual(set(crowding_distance(points).values()), {0.0})
        self.assertEqual(promote(points, 2), ['a','b'])
        self.assertEqual(promote(points, 6), ['a','b','c'])

    def test_refuse_mixed_or_partial_budgets(self):
        with self.assertRaises(ValueError):
            promote([self.obs('a',.8,100,10), self.obs('b',.8,100,20,30)],1)
        with self.assertRaises(ValueError):
            final_pareto([self.obs('a',.8,100,10)],60)

    def test_rung_uses_last_three_and_projects_time(self):
        history=[dict(epoch=i,validation_accuracy=i/100,training_seconds=i*2) for i in range(1,16)]
        observation=rung_observation('a',history,100,60)
        self.assertAlmostEqual(observation.accuracy,.14)
        self.assertEqual(observation.seconds,120)


class CohortTests(unittest.TestCase):
    def test_full_synchronous_schedule_with_resume_and_no_training(self):
        p=Protocol()
        cohort=Cohort(p,startup_candidates(12,42))
        jobs=cohort.jobs()
        with self.assertRaises(ValueError):cohort.advance()
        first=jobs[0][0]
        cohort.record(first.id,[dict(epoch=i,validation_accuracy=.5,training_seconds=i) for i in range(1,8)],100)
        self.assertEqual(len(cohort.jobs()),12)
        cohort=Cohort.restore(json.loads(json.dumps(cohort.state())))
        counts=[]
        for rung in p.rungs:
            jobs=cohort.jobs();counts.append(len(jobs))
            for c, target in jobs:
                self.assertEqual(target,rung)
                cohort.record(c.id,[dict(epoch=i,validation_accuracy=.5,training_seconds=i) for i in range(1,rung+1)],100)
            cohort.advance()
        self.assertEqual(counts,[12,6,3])
        self.assertEqual(len(cohort.completed()),3)
        self.assertFalse(cohort.jobs())
        self.assertTrue(all(o.epoch==60 for o in cohort.completed()))

    def test_startup_bypasses_pruning(self):
        cohort=Cohort(Protocol(),startup_candidates(12,42),startup=True)
        self.assertEqual({e for _,e in cohort.jobs()},{60})

    def test_all_failed_ends_cohort(self):
        cohort=Cohort(Protocol(),startup_candidates(12,42))
        for c in cohort.candidates:cohort.fail(c.id,'simulated failure')
        self.assertEqual(cohort.advance(),[])
        self.assertEqual(cohort.completed(),[])

    def test_fewer_survivors_than_quota(self):
        cohort=Cohort(Protocol(),startup_candidates(12,42))
        for c in cohort.candidates[2:]:cohort.fail(c.id,'simulated failure')
        for c,_ in cohort.jobs():
            cohort.record(c.id,[dict(epoch=i,validation_accuracy=.6,training_seconds=i) for i in range(1,16)],100)
        self.assertEqual(len(cohort.advance()),2)


class StorageTests(unittest.TestCase):
    def test_atomic_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';write_json(path,{'epoch':3})
            def broken(stream):
                stream.write(b'partial')
                raise OSError('simulated interruption')
            with self.assertRaises(OSError):atomic_write(path,broken)
            self.assertEqual(read_json(path),{'epoch':3})
            self.assertFalse(list(Path(temp).glob('*.tmp')))

    def test_run_lock_excludes_second_writer(self):
        with tempfile.TemporaryDirectory() as temp, run_lock(temp):
            with self.assertRaises(RuntimeError):
                with run_lock(temp):pass

    def test_checkpoint_and_random_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pt'
            state=random_states()
            save_checkpoint(path,dict(schema=1,identity='same',model={'weight':torch.arange(3)},random_states=state))
            checkpoint=load_checkpoint(path,'same')
            self.assertTrue(torch.equal(checkpoint['model']['weight'],torch.arange(3)))
            expected=(random.random(),float(np.random.random()),float(torch.rand(())))
            restore_random_states(checkpoint['random_states'])
            actual=(random.random(),float(np.random.random()),float(torch.rand(())))
            self.assertEqual(expected,actual)
            with self.assertRaises(ValueError):load_checkpoint(path,'changed')


class LedgerTests(unittest.TestCase):
    def test_archive_is_completed_only_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'study.sqlite';p=Protocol(active_search_seconds=100)
            ledger=Ledger(path,p,'data','machine')
            c=Candidate();ledger.add_cohort(0,[c])
            ledger.record_rung(Observation(c.id,.7,100,15,15))
            self.assertEqual(ledger.archive(),[])
            with self.assertRaises(ValueError):ledger.complete(Observation(c.id,.7,100,15,15),20)
            with self.assertRaises(ValueError):ledger.complete(Observation(c.id,.8,100,60,60),101)
            ledger.complete(Observation(c.id,.8,100,60,60),80)
            ledger.close()
            ledger=Ledger(path,p,'data','machine')
            self.assertEqual(len(ledger.archive()),1);self.assertEqual(len(ledger.pareto()),1)
            ledger.close()
            with self.assertRaises(ValueError):Ledger(path,p,'different data','machine')


class SamplerTests(unittest.TestCase):
    def test_startup_is_balanced_and_deterministic(self):
        configs=startup_candidates(12,42)
        self.assertEqual(sum(c.optimizer=='sgd' for c in configs),6)
        self.assertEqual(configs,startup_candidates(12,42))
        self.assertEqual(configs[0],Candidate())

    def test_motpe_replay_and_completed_only_history(self):
        configs=startup_candidates(12,42)
        history=[(c,Observation(c.id,.6+i*.01,1000+i*100,30+i,60)) for i,c in enumerate(configs)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a=propose(history,full_epochs=60,startup_trials=12,seed=12)
            b=propose(history,full_epochs=60,startup_trials=12,seed=12)
        self.assertEqual(a,b)
        self.assertNotIn(a.id,{c.id for c in configs})
        bad=[(configs[0],Observation(configs[0].id,.8,100,20,15))]
        with self.assertRaises(ValueError):propose(bad,full_epochs=60,startup_trials=12,seed=12)


class MetricsTests(unittest.TestCase):
    def test_exact_corruption_metric_units(self):
        scores={(c,s):.6 for c in CORRUPTIONS for s in range(1,6)}
        result=robustness_metrics(.8,scores)
        self.assertAlmostEqual(result['mean_corruption_error'],.4)
        self.assertAlmostEqual(result['accuracy_drop_percentage_points'],20)
        self.assertFalse(result['reference_normalized'])
        scores.pop(('fog',3))
        with self.assertRaises(ValueError):robustness_metrics(.8,scores)


class TrainerControlTests(unittest.TestCase):
    def test_pause_resume_control_with_simulated_epochs_only(self):
        # No optimizer step, dataset read, backward pass, or GPU execution here.
        p=Protocol(full_epochs=6,warmup_epochs=1,rungs=(3,6),survivors=(1,),cohort_size=2,startup_trials=2)
        c=Candidate(width=16)
        manifest={'id':'synthetic','classes':list(map(str,range(10)))}
        with tempfile.TemporaryDirectory() as temp, \
             patch('cs400.trainer.load_manifest',return_value=manifest), \
             patch('cs400.trainer.search_datasets',return_value=([0],[0])), \
             patch('cs400.trainer.train_epoch',return_value=(.7,1.5)) as fake_train, \
             patch('cs400.trainer.validation_accuracy',return_value=.6):
            result=train_candidate(c,p,'unused','unused',temp,target_epoch=3,session_seconds=100,device_name='cpu')
            self.assertEqual(result['status'],'target_reached')
            self.assertEqual(fake_train.call_count,3)
            result=train_candidate(c,p,'unused','unused',temp,target_epoch=6,session_seconds=100,device_name='cpu',resume=True)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(fake_train.call_count,6)
            self.assertEqual(result['training_seconds'],9)
            history=read_json(Path(temp)/'history.json')
            self.assertEqual([row['epoch'] for row in history],list(range(1,7)))
            self.assertAlmostEqual(history[-1]['learning_rate'],c.learning_rate*.01)
            # Recover when checkpoint commit succeeded but its JSON mirror was interrupted.
            write_json(Path(temp)/'history.json',history[:-1])
            train_candidate(c,p,'unused','unused',temp,target_epoch=6,session_seconds=100,device_name='cpu',resume=True)
            self.assertEqual(read_json(Path(temp)/'history.json'),history)
            self.assertEqual(fake_train.call_count,6)
            with self.assertRaises(ValueError):
                train_candidate(c,replace(p,batch_size=64),'unused','unused',temp,target_epoch=6,session_seconds=100,device_name='cpu',resume=True)

    def test_cli_requires_explicit_training_flag(self):
        from cs400.cli import parser, execute
        args=parser().parse_args(['train','--run-dir','unused','--target-epoch','3','--session-minutes','15'])
        with patch('cs400.trainer.train_candidate') as trainer, self.assertRaises(ValueError):execute(args)
        trainer.assert_not_called()


if __name__=='__main__':
    unittest.main()
