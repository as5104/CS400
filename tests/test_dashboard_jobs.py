"""Upload and job integration tests. Training/evaluation are simulated, never optimized."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch,Mock
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import zipfile
import numpy as np
from PIL import Image
from cs400.uploads import import_zip
from cs400.local_jobs import LocalJobs,execute_job,settings
from cs400.storage import read_json,write_json
from cs400.search import search_session
from cs400.dashboard import make_server


def archive_fixture(path,explicit=False):
    rng=np.random.default_rng(7)
    with zipfile.ZipFile(path,'w') as z:
        for label in ('alpha','beta'):
            for i in range(20):
                data=io.BytesIO();Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(data,format='PNG')
                split=('train' if i<14 else 'validation' if i<17 else 'test')+'/' if explicit else ''
                z.writestr(f'dataset/{split}{label}/{i}.png',data.getvalue())


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.archive=self.root/'images.zip';archive_fixture(self.archive)
    def tearDown(self):self.temp.cleanup()
    def test_automatic_split_is_persisted_and_reproducible(self):
        a=import_zip(self.root,self.archive,augmentation_confirmed=True)
        b=import_zip(self.root,self.archive,augmentation_confirmed=True)
        self.assertEqual(a['counts'],{'train':28,'validation':6,'test':6})
        self.assertEqual(read_json(self.root/a['manifest']),read_json(self.root/b['manifest']))
    def test_explicit_splits_preserved(self):
        archive_fixture(self.archive,True);d=import_zip(self.root,self.archive,augmentation_confirmed=True)
        self.assertEqual(d['split_policy'],'provided splits');self.assertEqual(d['counts']['test'],6)
    def test_unsafe_path_rejected_without_publishing(self):
        with zipfile.ZipFile(self.archive,'a') as z:z.writestr('../escape.png',b'bad')
        with self.assertRaises(ValueError):import_zip(self.root,self.archive,augmentation_confirmed=True)
        self.assertFalse((self.root/'escape.png').exists());self.assertEqual(list((self.root/'data/uploads').glob('*/dataset.json')),[])
    def test_duplicate_pixels_rejected(self):
        with zipfile.ZipFile(self.archive,'a') as z:z.writestr('dataset/beta/copy.png',z.read('dataset/alpha/0.png'))
        with self.assertRaisesRegex(ValueError,'distinct decoded'):import_zip(self.root,self.archive,augmentation_confirmed=True)
    def test_unsupported_files_and_expanded_limit(self):
        with patch('cs400.uploads.MAX_EXPANDED',100),self.assertRaisesRegex(ValueError,'expanded limit'):
            import_zip(self.root,self.archive,augmentation_confirmed=True)
        with zipfile.ZipFile(self.archive,'a') as z:z.writestr('dataset/notes.txt','not an image')
        with self.assertRaisesRegex(ValueError,'Only PNG/JPEG'):import_zip(self.root,self.archive,augmentation_confirmed=True)
    def test_confirmation_and_corrupt_image(self):
        with self.assertRaises(ValueError):import_zip(self.root,self.archive)
        with zipfile.ZipFile(self.archive,'a') as z:z.writestr('dataset/alpha/broken.png',b'not png')
        with self.assertRaises(OSError):import_zip(self.root,self.archive,augmentation_confirmed=True)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);archive=self.root/'images.zip';archive_fixture(archive)
        self.data=import_zip(self.root,archive,augmentation_confirmed=True);self.manager=LocalJobs(self.root)
        self.job=self.manager.create(self.data['id'],dict(epochs=12,budget_hours=.05,session_minutes=5,device='cpu'))
    def tearDown(self):self.temp.cleanup()
    def pilot(self,candidate,protocol,data,manifest,directory,**kwargs):
        directory=Path(directory);write_json(directory/'sessions.json',[{'active_seconds':3}]);write_json(directory/'run.json',{})
        write_json(directory/'history.json',[{'epoch':i} for i in (1,2,3)])
        return {'epoch':3,'accounting_uncertain':False}
    def test_create_does_not_launch_and_requires_explicit_start(self):
        self.assertEqual(self.job['status'],'ready')
        with patch('cs400.local_jobs.subprocess.Popen') as popen:
            with self.assertRaises(ValueError):self.manager.launch(self.job['id'])
            popen.assert_not_called()
            proc=Mock();proc.poll.return_value=None;popen.return_value=proc
            self.manager.launch(self.job['id'],start_training=True)
            self.assertIn('cs400.local_worker',popen.call_args.args[0]);self.manager.pause(self.job['id'])
            self.assertTrue((self.manager.path(self.job['id'])/'pause.request').exists())
            other=self.manager.create(self.data['id'],{'device':'cpu'})
            with self.assertRaisesRegex(ValueError,'Another local job'):self.manager.launch(other['id'],start_training=True)
    def test_invalid_settings(self):
        for invalid in ({'epochs':13},{'budget_hours':float('nan')},{'device':'auto'},{'batch_size':True},{'shell':'bad'}):
            with self.assertRaises(ValueError):settings(invalid)
    def test_calibration_budget_guard_and_revision(self):
        def slow(*a,**k):
            value=self.pilot(*a,**k);write_json(Path(a[4])/'sessions.json',[{'active_seconds':30}]);return value
        search=Mock();execute_job(self.root,self.job['id'],train_fn=slow,search_fn=search)
        result=self.manager.status(self.job['id']);self.assertEqual(result['status'],'needs_budget');search.assert_not_called()
        self.manager.revise_budget(self.job['id'],1)
        self.assertEqual(self.manager.status(self.job['id'])['protocol']['active_search_seconds'],3600)
    def test_pause_after_calibration_prevents_search(self):
        (self.manager.path(self.job['id'])/'pause.request').touch()
        search=Mock();execute_job(self.root,self.job['id'],train_fn=self.pilot,search_fn=search)
        search.assert_not_called();self.assertEqual(self.manager.status(self.job['id'])['status'],'paused')
    def test_real_controller_to_frozen_results_with_simulated_epochs(self):
        now=[0.0];calls=[]
        def fake_train(candidate,protocol,data,manifest,directory,target_epoch,**kwargs):
            directory=Path(directory);history_path=directory/'history.json'
            history=read_json(history_path) if history_path.exists() else []
            old=len(history)
            for epoch in range(old+1,target_epoch+1):history.append({'epoch':epoch,'validation_accuracy':.7,'training_seconds':float(epoch)*2})
            write_json(directory/'history.json',history);write_json(directory/'run.json',{})
            now[0]+=(target_epoch-old)*2;calls.append((old,target_epoch))
            return dict(epoch=target_epoch,parameters=100,validation_accuracy=.7,training_seconds=target_epoch*2.,status='target_reached',accounting_uncertain=False)
        def search(protocol,data,manifest,directory,**kwargs):
            return search_session(protocol,data,manifest,directory,**kwargs,train_fn=fake_train,clock=lambda:now[0])
        evaluated=[];exported=[]
        def evaluate(trial,manifest,data,out,device):
            self.assertTrue(read_json(Path(trial).parent.parent/'search-results.json')['frozen']);evaluated.append(trial)
            write_json(out,{'metrics':{'clean_accuracy':.6}})
        def export(trial,manifest,out):
            exported.append(trial);Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_bytes(b'test fixture, not weights')
        with patch('cs400.search.environment',return_value={'device':'fake-cpu'}):
            execute_job(self.root,self.job['id'],train_fn=self.pilot,search_fn=search,evaluate_fn=evaluate,export_fn=export)
        job=self.manager.status(self.job['id']);self.assertEqual(job['status'],'complete')
        result=read_json(self.root/'runs'/job['run_name']/'search-results.json')
        self.assertGreaterEqual(len(result['completed']),5);self.assertEqual(len(evaluated),len(result['completed']))
        self.assertEqual(len(exported),len(result['pareto']));self.assertTrue(any(a==3 and b==6 for a,b in calls))
        with self.assertRaises(ValueError):self.manager.revise_budget(self.job['id'],2)
    def test_evaluation_resumes_without_repeating_search(self):
        path=self.manager.path(self.job['id']);job=read_json(path/'job.json');job.update(calibration={'estimated_epoch_seconds':1},phase='evaluation',status='evaluation_failed');write_json(path/'job.json',job)
        run=self.root/'runs'/job['run_name'];cid='a'*64
        write_json(run/'search-results.json',{'frozen':True,'completed':[{'metrics':{'candidate_id':cid}}],'pareto':[{'candidate_id':cid}],'stop_reason':'active_budget_reached'})
        def evaluate(trial,manifest,data,out,device):write_json(out,{})
        def export(trial,manifest,out):Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_bytes(b'fixture')
        search=Mock();train=Mock()
        execute_job(self.root,self.job['id'],train_fn=train,search_fn=search,evaluate_fn=evaluate,export_fn=export)
        search.assert_not_called();train.assert_not_called();self.assertEqual(self.manager.status(job['id'])['status'],'complete')


class UploadHTTPTests(unittest.TestCase):
    def test_real_upload_and_create_routes_never_start_training(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'images.zip';archive_fixture(archive)
            server=make_server(temp,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            address=f'http://127.0.0.1:{server.server_address[1]}'
            try:
                headers={'Content-Type':'application/zip','Origin':address,'X-Augmentation-Confirmed':'true'}
                with urlopen(Request(address+'/api/upload',data=archive.read_bytes(),headers=headers)) as response:dataset=json.load(response)
                with patch('cs400.local_jobs.subprocess.Popen') as process:
                    request={'dataset_id':dataset['id'],'settings':{'device':'cpu'}}
                    headers={'Content-Type':'application/json','Origin':address}
                    with urlopen(Request(address+'/api/jobs/create',data=json.dumps(request).encode(),headers=headers)) as response:job=json.load(response)
                    with self.assertRaises(HTTPError):urlopen(Request(address+'/api/jobs/start',data=json.dumps({'id':job['id']}).encode(),headers=headers))
                    process.assert_not_called()
            finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
