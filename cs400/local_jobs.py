"""Persisted local jobs: explicit start, isolated worker, checkpointed pause."""
from dataclasses import asdict, replace
from contextlib import closing
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from .config import Protocol
from .storage import read_json, write_json, run_lock

BUSY={'queued','calibrating','searching','evaluating'}


def hardware():
    import torch
    return dict(cuda_available=torch.cuda.is_available(),device='cuda' if torch.cuda.is_available() else 'cpu',
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def settings(value):
    allowed={'epochs','batch_size','budget_hours','session_minutes','seed','device'}
    if set(value)-allowed:raise ValueError('Unsupported search setting')
    def integer(key,default,lo,hi):
        n=value.get(key,default)
        if type(n) is not int or not lo<=n<=hi:raise ValueError(f'{key} must be an integer from {lo} to {hi}')
        return n
    epochs=integer('epochs',60,12,120)
    if epochs%4:raise ValueError('Epoch count must be a multiple of four')
    batch=integer('batch_size',64,8,128)
    seed=integer('seed',42,0,2**32-121)
    def number(key,default,lo,hi):
        n=value.get(key,default)
        if type(n) not in (int,float) or not math.isfinite(n) or not lo<=n<=hi:raise ValueError(f'{key} must be between {lo} and {hi}')
        return float(n)
    budget=number('budget_hours',5.25,.05,48)
    session=number('session_minutes',30,1,240)
    device=value.get('device','cuda')
    if device not in ('cuda','cpu'):raise ValueError('Select CPU or CUDA')
    protocol=Protocol(full_epochs=epochs,warmup_epochs=min(5,epochs//4),batch_size=batch,
                      seed=seed,rungs=(epochs//4,epochs//2,epochs),survivors=(2,1),cohort_size=4,
                      startup_trials=4,active_search_seconds=budget*3600,pilot_confirmed=False)
    return protocol,session,device


class LocalJobs:
    def __init__(self,root):
        self.root=Path(root).resolve();self.directory=self.root/'workspace/jobs'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.control=threading.Lock();self.processes={}

    def path(self,uid):
        if not isinstance(uid,str) or not re.fullmatch('[0-9a-f]{32}',uid):raise ValueError('Invalid job identifier')
        path=self.directory/uid
        if not (path/'job.json').is_file():raise ValueError('Job not found')
        return path

    def dataset(self,uid):
        if not isinstance(uid,str) or not re.fullmatch('[0-9a-f]{32}',uid):raise ValueError('Invalid dataset identifier')
        return read_json(self.root/'data/uploads'/uid/'dataset.json')

    def create(self,dataset_id,options):
        dataset=self.dataset(dataset_id);protocol,session,device=settings(options)
        if device=='cuda' and not hardware()['cuda_available']:raise ValueError('CUDA is unavailable. Select CPU explicitly')
        uid=uuid.uuid4().hex;path=self.directory/uid;path.mkdir()
        job=dict(id=uid,dataset=dataset,protocol=asdict(protocol),session_minutes=session,device=device,
                 run_name='local-'+uid,created=time.time(),status='ready',phase='calibration',
                 message='Ready. Start runs a three-epoch calibration, then MOTPE with multi-fidelity search.')
        write_json(path/'job.json',job)
        return job

    def running(self,job):
        process=self.processes.get(job['id'])
        if process is not None:return process.poll() is None
        owner=self.root/'workspace/compute/owner.json'
        try:
            with run_lock(self.root/'workspace/compute'):return False
        except RuntimeError:
            return owner.exists() and read_json(owner).get('id')==job['id']

    def launch(self,uid,*,start_training=False):
        if start_training is not True:raise ValueError('Explicit Start / Resume acknowledgement is required')
        with self.control:
            path=self.path(uid);job=read_json(path/'job.json')
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError('Another local job is running. Pause it before starting a second job')
            if self.running(job):raise ValueError('This job is already running')
            if job['status'] not in ('ready','paused','needs_budget','evaluation_failed'):
                raise ValueError('This job cannot resume; inspect its status or create a new search')
            with run_lock(self.root/'workspace/compute'):
                pass  # worker holds this lock for its entire GPU/CPU session
            (path/'pause.request').unlink(missing_ok=True)
            previous=job['status'];job.update(status='queued',message='Starting local worker',queued=time.time())
            write_json(path/'job.json',job)
            try:
                with (path/'worker.log').open('ab') as log:
                    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)
                    self.processes[uid]=subprocess.Popen([sys.executable,'-m','cs400.local_worker','--root',str(self.root),'--job',uid],
                        cwd=self.root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            except Exception:
                job.update(status=previous,message='Worker could not start');write_json(path/'job.json',job);raise
        return self.status(uid)

    def pause(self,uid):
        path=self.path(uid);job=read_json(path/'job.json')
        if not self.running(job):raise ValueError('Job is not running')
        (path/'pause.request').touch()
        return dict(message='Pause requested. Wait for the current epoch or evaluation to finish and save.')

    def revise_budget(self,uid,hours):
        with self.control:
            path=self.path(uid);job=read_json(path/'job.json')
            if self.running(job) or job['status']!='needs_budget':raise ValueError('Budget can change only before the search begins')
            run=self.root/'runs'/job['run_name']
            if (run/'search.sqlite').exists():raise ValueError('Search protocol is already frozen')
            if type(hours) not in (int,float) or not math.isfinite(hours) or not .05<=hours<=48:raise ValueError('Budget must be 0.05 to 48 hours')
            job['protocol']['active_search_seconds']=hours*3600
            write_json(path/'job.json',job)
        return self.status(uid)

    def status(self,uid):
        path=self.path(uid);job=read_json(path/'job.json');alive=self.running(job)
        # Never automatically resume an unexpectedly interrupted controlled search.
        if job['status'] in BUSY and not alive and time.time()-job.get('queued',0)>5:
            job={**job,'status':'interrupted','message':'Worker stopped unexpectedly. Checkpoints are retained; create a new search rather than silently changing time accounting.'}
        result={**job,'running':alive,'pause_requested':(path/'pause.request').exists()}
        result['results_available']=(self.root/'runs'/job['run_name']/'search-results.json').exists()
        log=path/'worker.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-12000));result['log']=f.read().decode('utf-8',errors='replace')
        trials=self.root/'runs'/job['run_name']/'trials'
        histories=[]
        for history in trials.glob('*/history.json'):
            rows=read_json(history)
            if rows:histories.append(dict(candidate=history.parent.name[:12],**rows[-1],updated_at_ns=history.stat().st_mtime_ns))
        histories.sort(key=lambda row: (-row['updated_at_ns'], row['candidate']))
        result['trials']=histories
        pilot=path/'calibration/history.json'
        result['calibration_epochs']=len(read_json(pilot)) if pilot.exists() else 0
        db=trials.parent/'search.sqlite'
        if db.exists():
            try:
                with closing(sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True)) as conn:
                    result['completed']=conn.execute('SELECT COUNT(*) FROM completed').fetchone()[0]
                    state=conn.execute("SELECT value FROM metadata WHERE key='controller'").fetchone()
                    if state:result['active_seconds']=json.loads(state[0])['active_seconds']
            except sqlite3.OperationalError:pass
        return result

    def list(self):
        return [self.status(p.parent.name) for p in sorted(self.directory.glob('*/job.json'),reverse=True)]


def execute_job(root,uid,*,train_fn=None,search_fn=None,evaluate_fn=None,export_fn=None):
    """Worker entry. Dependency injection is used only by tests; no training on import."""
    from .trainer import train_candidate
    from .search import search_session, cohort_work
    from .evaluation import evaluate_clean,export_run
    from .config import Candidate
    train_fn=train_fn or train_candidate;search_fn=search_fn or search_session
    evaluate_fn=evaluate_fn or evaluate_clean;export_fn=export_fn or export_run
    manager=LocalJobs(root);path=manager.path(uid);job=read_json(path/'job.json')
    pause=lambda:(path/'pause.request').exists()
    started=time.perf_counter();session=job['session_minutes']*60
    def save(status,message,**extra):
        job.update(status=status,message=message,**extra);write_json(path/'job.json',job)
    try:
        with run_lock(manager.root/'workspace/compute'):
            write_json(manager.root/'workspace/compute/owner.json',{'id':uid})
            protocol=Protocol(**job['protocol']);data=job['dataset']
            data_root=manager.root/data['data_root'];manifest=manager.root/data['manifest']
            run=manager.root/'runs'/job['run_name']
            if not job.get('calibration'):
                save('calibrating','Checking the largest supported ResNet with AdamW for three epochs.',phase='calibration')
                candidate=Candidate(depth='3-3-3-3',width=32,dropout=.2,augmentation='erasing')
                result=train_fn(candidate,protocol,data_root,manifest,path/'calibration',target_epoch=3,
                                session_seconds=max(.01,session-(time.perf_counter()-started)),device_name=job['device'],
                                resume=(path/'calibration/run.json').exists(),pause_check=pause)
                if result.get('accounting_uncertain'):raise RuntimeError('Calibration ended unexpectedly; start a new job')
                if result['epoch']<3:
                    save('paused','Calibration paused; resume to finish the three-epoch check.');return
                sessions=read_json(path/'calibration/sessions.json')
                seconds=sum(s['active_seconds'] for s in sessions)
                estimate=max(.01,seconds/3*1.25)
                job['calibration']=dict(estimated_epoch_seconds=estimate,active_seconds=seconds,
                                       note='Three-epoch largest-model pilot with 25% allowance; not a guarantee of sustained speed')
                write_json(path/'job.json',job)
            protocol=replace(Protocol(**job['protocol']),pilot_confirmed=True)
            estimate=job['calibration']['estimated_epoch_seconds']
            minimum=(protocol.startup_trials*protocol.full_epochs+cohort_work(protocol,True))*estimate
            if not (run/'search.sqlite').exists() and minimum>protocol.active_search_seconds:
                save('needs_budget',f'Calibration estimates at least {minimum/3600:.2f} search hours for startup and one cohort. Increase the budget, or create a shorter-epoch job.',minimum_hours=minimum/3600);return
            report_path=run/'search-results.json'
            frozen=report_path.exists() and read_json(report_path)['frozen']
            if not frozen:
                if pause() or time.perf_counter()-started>=session:
                    save('paused','Calibration complete. Resume to start or continue search.');return
                save('searching','MOTPE is proposing configurations; candidates train at equal-budget rungs.',phase='search')
                write_json(path/'protocol.json',asdict(protocol))
                report=search_fn(protocol,data_root,manifest,run,
                                 session_seconds=max(.01,session-(time.perf_counter()-started)),
                                 estimated_epoch_seconds=estimate,method='motpe',pruning=True,device_name=job['device'],
                                 resume=(run/'search.sqlite').exists(),pause_check=pause)
                if not report['frozen']:
                    save('paused','Session paused and checkpoints saved. Resume when ready.');return
            else:report=read_json(report_path)
            if report['stop_reason']=='startup_incomplete_inspect_failures':
                raise RuntimeError('A startup candidate failed. Inspect the log; use a smaller batch in a new job if memory was exhausted.')
            if not report['completed']:
                save('finished_empty','Budget ended before an eligible model completed. Increase the allocation in a new job.');return
            save('evaluating','Search is frozen. Evaluating clean test accuracy; test results never change Pareto selection.',phase='evaluation')
            for entry in report['completed']:
                if pause() or time.perf_counter()-started>=session:
                    save('paused','Search is frozen. Resume to finish post-search evaluation and exports.');return
                cid=entry['metrics']['candidate_id'];out=run/'evaluations'/f'{cid}.json'
                if not out.exists():evaluate_fn(run/'trials'/cid,manifest,data_root,out,job['device'])
            for entry in report['pareto']:
                if pause() or time.perf_counter()-started>=session:
                    save('paused','Clean test evaluation saved. Resume to finish Pareto model exports.');return
                cid=entry['candidate_id'];out=run/'exports'/f'{cid}.pt'
                if not out.exists():export_fn(run/'trials'/cid,manifest,out)
            save('complete',f"Finished: {len(report['completed'])} completed models and {len(report['pareto'])} observed Pareto models. Custom corruption scores are not measured.",phase='complete')
    except Exception as exc:
        save('evaluation_failed' if job.get('phase')=='evaluation' else 'failed',f'{type(exc).__name__}: {exc}')
        raise
