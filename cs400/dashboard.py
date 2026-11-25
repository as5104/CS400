"""Loopback-only dataset upload, explicit local search jobs and results interface."""
import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
import zipfile
from urllib.parse import parse_qs, urlparse
from .storage import read_json


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.operation_lock = threading.Lock()
        from .local_jobs import LocalJobs
        self.jobs = LocalJobs(self.root)

    def run_directory(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid run name")
        directory = (self.root / "runs" / name).resolve()
        if not directory.is_relative_to(self.root / "runs") or not (directory / "search-results.json").is_file():
            raise ValueError("Search not found")
        return directory

    def runs(self):
        rows = []
        for path in sorted((self.root / "runs").glob("*/search-results.json")):
            try:
                directory = self.run_directory(path.parent.name)
                report = read_json(directory / "search-results.json")
                rows.append(dict(name=directory.name, frozen=report["frozen"], completed=len(report["completed"]),
                                 pareto=len(report["pareto"]), active_seconds=report["active_seconds"]))
            except (OSError, ValueError, KeyError):
                continue
        return rows

    def detail(self, name):
        directory = self.run_directory(name)
        report = read_json(directory / "search-results.json")
        if report["frozen"]:
            from .comparison import load_search
            load_search(directory)  # verify displayed archive against the saved ledger
        evaluated = {}
        for path in (directory / "evaluations").glob("*.json"):
            item = read_json(path)
            from .config import Candidate
            evaluated[Candidate(**item["candidate"]).id] = item
        models = []
        front = {row["candidate_id"] for row in report["pareto"]}
        for row in report["completed"]:
            cid = row["metrics"]["candidate_id"]
            history_path = directory / "trials" / cid / "history.json"
            models.append(dict(candidate=row["candidate"], metrics=row["metrics"], pareto=cid in front,
                               evaluation=evaluated.get(cid),
                               history=read_json(history_path) if history_path.exists() else []))
        progress = []
        for path in sorted((directory / "trials").glob("*/result.json")):
            value = read_json(path)
            progress.append({k: value.get(k) for k in ("candidate_id", "epoch", "full_epochs", "status", "validation_accuracy")})
        manifest_path='data/cifar10-split.json'
        dataset_name='CIFAR-10'
        if name.startswith('local-'):
            job=self.jobs.status(name[6:])
            manifest_path=job['dataset']['manifest'];dataset_name='Uploaded dataset ('+str(len(job['dataset']['classes']))+' classes)'
        return dict(name=name, frozen=report["frozen"], stop_reason=report["stop_reason"],
                    active_seconds=report["active_seconds"], protocol=report["protocol"], models=models,
                    progress=progress, manifest_id=report["manifest_id"],manifest_path=manifest_path,dataset_name=dataset_name)

    def export(self, name, cid, manifest_name):
        directory = self.run_directory(name)
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise ValueError("Invalid model identifier")
        manifest = (self.root / manifest_name).resolve()
        if not manifest.is_relative_to(self.root / "data") or manifest.suffix != ".json":
            raise ValueError("Select a manifest JSON inside this project's data directory")
        target = directory / "exports" / f"{cid}.pt"
        from .evaluation import completed_model, export_run
        if not target.exists():
            export_run(directory / "trials" / cid, manifest, target)
        else:
            # Still check archive eligibility and the supplied manifest.
            completed_model(directory / "trials" / cid, manifest, "cpu")
        return target

    def download(self, relative):
        path = (self.root / relative).resolve()
        in_results = path.is_relative_to(self.root / "output" / "final-results")
        in_exports = path.is_relative_to(self.root / "runs") and path.parent.name == "exports" and path.suffix == ".pt"
        in_demo = path == self.root / 'output/demo/synthetic-shapes-demo.zip'
        if not (in_results or in_exports or in_demo) or not path.is_file():
            raise ValueError("Download is unavailable")
        return path


def make_server(root, port=8765):
    workspace = Workspace(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def allowed_host(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def send(self, status, data, content_type="application/json"):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.allowed_host():
                return self.send(403, {"error": "Loopback host required"})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.send(200, Path(__file__).with_name("dashboard.html").read_bytes(), "text/html; charset=utf-8")
                elif parsed.path == "/api/runs":
                    self.send(200, workspace.runs())
                elif parsed.path == "/api/run":
                    self.send(200, workspace.detail(query.get("name", [""])[0]))
                elif parsed.path == "/api/evidence":
                    path = workspace.root / "output/final-results/summary.json"
                    self.send(200, read_json(path) if path.exists() else {})
                elif parsed.path == '/api/hardware':
                    from .local_jobs import hardware
                    self.send(200,hardware())
                elif parsed.path == '/api/jobs':
                    self.send(200,workspace.jobs.list())
                elif parsed.path == '/api/job':
                    self.send(200,workspace.jobs.status(query.get('id',[''])[0]))
                elif parsed.path == '/api/datasets':
                    self.send(200,[read_json(p) for p in sorted((workspace.root/'data/uploads').glob('*/dataset.json'))])
                elif parsed.path == '/api/results.csv':
                    import csv,io
                    detail=workspace.detail(query.get('name',[''])[0])
                    if not detail['frozen']:raise ValueError('Final CSV is available after the search is frozen')
                    stream=io.StringIO();writer=csv.writer(stream)
                    writer.writerow(['candidate_id','depth','width','optimizer','learning_rate','weight_decay','dropout','augmentation','validation_accuracy','parameters','training_seconds','pareto','clean_test_accuracy'])
                    for model in detail['models']:
                        candidate=model['candidate'];metrics=model['metrics']
                        writer.writerow([metrics['candidate_id'],*[candidate[k] for k in ('depth','width','optimizer','learning_rate','weight_decay','dropout','augmentation')],metrics['accuracy'],metrics['parameters'],metrics['seconds'],model['pareto'],(model['evaluation'] or {}).get('metrics',{}).get('clean_accuracy','')])
                    self.send(200,stream.getvalue().encode(),'text/csv; charset=utf-8')
                elif parsed.path == "/download":
                    path = workspace.download(query.get("path", [""])[0])
                    self.send(200, path.read_bytes(), "application/octet-stream")
                else:
                    self.send(404, {"error": "Not found"})
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                self.send(400, {"error": str(exc)})

        def do_POST(self):
            port = self.server.server_address[1]
            if not self.allowed_host() or self.headers.get("Origin") not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self.send(403, {"error": "Same-origin local action required"})
            if urlparse(self.path).path == '/api/upload':
                return self.upload()
            if self.headers.get("Content-Type") != "application/json":
                return self.send(415, {"error": "JSON required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 12_000_000:
                    raise ValueError("Request must be at most 12 MB")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request,dict):raise ValueError('Expected a JSON object')
                if self.path == '/api/jobs/create':
                    return self.send(200,workspace.jobs.create(request['dataset_id'],request.get('settings',{})))
                if self.path == '/api/jobs/start':
                    return self.send(200,workspace.jobs.launch(request['id'],start_training=request.get('start_training')))
                if self.path == '/api/jobs/pause':
                    return self.send(200,workspace.jobs.pause(request['id']))
                if self.path == '/api/jobs/budget':
                    return self.send(200,workspace.jobs.revise_budget(request['id'],request['hours']))
                if self.path not in ("/api/export", "/api/predict"):
                    return self.send(404, {"error": "Not found"})
                if not workspace.operation_lock.acquire(blocking=False):
                    return self.send(409, {"error": "Another model operation is running"})
                try:
                    path = workspace.export(request["run"], request["candidate_id"], request.get("manifest", "data/cifar10-split.json"))
                    if self.path == "/api/export":
                        self.send(200, dict(path=path.relative_to(workspace.root).as_posix()))
                    else:
                        from .inference import predict_image
                        data = base64.b64decode(request["image"], validate=True)
                        with tempfile.TemporaryDirectory() as temp:
                            image_path = Path(temp) / "input.png"
                            image_path.write_bytes(data)
                            self.send(200, predict_image(path, image_path))
                finally:
                    workspace.operation_lock.release()
            except (ValueError, OSError, KeyError, RuntimeError, TypeError, binascii.Error) as exc:
                self.send(400, {"error": str(exc)})

        def upload(self):
            from .uploads import import_zip,MAX_UPLOAD
            try:
                if self.headers.get('Content-Type') not in ('application/zip','application/octet-stream'):
                    raise ValueError('Upload a ZIP archive')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=MAX_UPLOAD:raise ValueError('ZIP upload must be at most 512 MiB')
                if self.headers.get('X-Augmentation-Confirmed')!='true':raise ValueError('Confirm the dataset requirements first')
                incoming=workspace.root/'workspace/incoming';incoming.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(dir=incoming) as temp:
                    path=Path(temp)/'upload.zip'
                    with path.open('wb') as stream:
                        remaining=length
                        while remaining:
                            chunk=self.rfile.read(min(1024**2,remaining))
                            if not chunk:raise ValueError('Upload was interrupted')
                            stream.write(chunk);remaining-=len(chunk)
                    self.send(200,import_zip(workspace.root,path,augmentation_confirmed=True))
            except (ValueError,OSError,RuntimeError,zipfile.BadZipFile,UserWarning) as exc:
                self.send(400,{'error':str(exc)})

    server=ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.workspace=workspace
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.root, args.port)
    print(f"Dashboard: http://127.0.0.1:{server.server_address[1]} (Ctrl+C requests a pause and stops the server). Training starts only when you press Start / Resume.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in server.workspace.jobs.list():
            if job['running']:server.workspace.jobs.pause(job['id'])
        server.server_close()


if __name__ == "__main__":
    main()
