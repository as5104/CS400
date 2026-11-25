"""Explicitly invoked single-candidate training, with epoch-boundary recovery."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import random
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import Candidate, Protocol, SCHEMA_VERSION, digest
from .data import load_manifest, search_datasets
from .model import ResNet, make_optimizer, parameter_count
from .storage import atomic_write, read_json, run_lock, write_json


def source_fingerprint():
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode())
        source_hash.update(source.read_bytes())
    return source_hash.hexdigest()


def environment(device):
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torchvision=importlib.metadata.version("torchvision"),
                source_fingerprint=source_fingerprint(),
                numpy=str(np.__version__), os=platform.platform(), machine=platform.node(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def random_states():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_states(states):
    random.setstate(states["python"])
    value = states["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(states["torch"].cpu())
    if states["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA checkpoint must resume on its recorded device")
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])


def save_checkpoint(path, payload):
    atomic_write(path, lambda stream: torch.save(payload, stream))


def load_checkpoint(path, expected_identity=None):
    # Our own checkpoints contain tensors and primitive containers only.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    if expected_identity is not None and value.get("identity") != expected_identity:
        raise ValueError("Checkpoint configuration, protocol, data, or hardware does not match")
    return value


@contextmanager
def pause_signals():
    requested = [False]
    def request_pause(signum, frame):
        requested[0] = True
        print("Pause requested. Finishing this epoch and saving its checkpoint.", flush=True)
    previous = signal.signal(signal.SIGINT, request_pause)
    try:
        yield requested
    finally:
        signal.signal(signal.SIGINT, previous)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def loader(dataset, protocol, seed, shuffle, device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=protocol.batch_size, shuffle=shuffle,
                      num_workers=protocol.workers, generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=False,
                      pin_memory=device.type == "cuda")


def validation_accuracy(model, batches, device):
    model.eval()
    correct = count = 0
    with torch.inference_mode():
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite validation outputs")
            correct += int((scores.argmax(1) == labels).sum().item())
            count += len(labels)
    if not count:
        raise ValueError("Validation split is empty")
    return correct / count


def train_epoch(model, batches, optimizer, scaler, device, amp):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = count = 0
    synchronize(device)
    started = time.perf_counter()
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; candidate cannot be a completed result")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().item()) * len(labels)
        count += len(labels)
    synchronize(device)
    elapsed = time.perf_counter() - started
    if not count:
        raise ValueError("Training split is empty")
    return loss_sum / count, elapsed


def train_candidate(candidate, protocol, data_root, manifest_path, run_dir, *,
                    target_epoch, session_seconds, device_name="cuda", resume=False, pause_check=None):
    """Never called by preflight or unit tests. The CLI requires --start-training."""
    if type(target_epoch) is not int or not 1 <= target_epoch <= protocol.full_epochs:
        raise ValueError("Target must be between 1 and full_epochs")
    if not math.isfinite(session_seconds) or session_seconds <= 0:
        raise ValueError("Specify a finite positive session limit")
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Initial trainer supports cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose CPU explicitly rather than silently changing hardware")
    if protocol.precision == "amp" and device.type != "cuda":
        raise ValueError("Initial AMP path requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(protocol.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = protocol.deterministic
    manifest = load_manifest(manifest_path)
    metadata = dict(schema=SCHEMA_VERSION, candidate=asdict(candidate), protocol=asdict(protocol),
                    manifest_id=manifest["id"], environment=environment(device),
                    seed_policy="trial seed + epoch; fresh nonpersistent loader each epoch")
    identity = digest(metadata)
    directory = Path(run_dir)
    checkpoint_path = directory / "checkpoint.pt"
    with run_lock(directory), pause_signals() as requested:
        meta_path = directory / "run.json"
        session_log_path = directory / "sessions.json"
        if meta_path.exists():
            if not resume:
                raise FileExistsError("Run exists. Use --resume or choose a new directory")
            if digest(read_json(meta_path)) != identity:
                raise ValueError("Run settings changed; create a separate run")
        elif resume:
            raise FileNotFoundError("No run metadata to resume")
        else:
            write_json(meta_path, metadata)
        sessions = read_json(session_log_path) if session_log_path.exists() else []
        uncertain = any(s["status"] == "running" for s in sessions)
        if uncertain:
            print("Previous session ended unexpectedly; its uncommitted work has unknown cost.", flush=True)
        session = dict(status="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       requested_seconds=session_seconds, accounting_uncertain=uncertain)
        sessions.append(session)
        write_json(session_log_path, sessions)
        started = time.perf_counter()
        status = "failed"
        try:
            seed_all(protocol.seed)
            model = ResNet(candidate, len(manifest["classes"]))
            parameters = parameter_count(model)
            if parameters > protocol.max_parameters:
                raise ValueError("Model exceeds the study parameter cap")
            model.to(device)
            optimizer = make_optimizer(model, candidate)
            scaler = torch.amp.GradScaler("cuda", enabled=protocol.precision == "amp")
            history = []
            if resume and checkpoint_path.exists():
                checkpoint = load_checkpoint(checkpoint_path, identity)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
                history = checkpoint["history"]
                restore_random_states(checkpoint["random_states"])
            completed = history[-1]["epoch"] if history else 0
            if completed > target_epoch:
                raise ValueError("Target precedes the saved checkpoint; use a new run for a different experiment")
            training, validation = search_datasets(data_root, manifest, candidate.augmentation)
            total_training_seconds = history[-1]["training_seconds"] if history else 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for epoch in range(completed + 1, target_epoch + 1):
                if pause_check is not None and pause_check():
                    requested[0] = True
                if requested[0] or time.perf_counter() - started >= session_seconds:
                    break
                # Epoch seeding makes worker/shuffle recovery independent of process lifetime.
                epoch_seed = protocol.seed + epoch
                seed_all(epoch_seed)
                lr = protocol.learning_rate(candidate.learning_rate, epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                train_batches = loader(training, protocol, epoch_seed, True, device)
                loss, seconds = train_epoch(model, train_batches, optimizer, scaler, device, protocol.precision == "amp")
                accuracy = validation_accuracy(model, loader(validation, protocol, epoch_seed, False, device), device)
                total_training_seconds += seconds
                history.append(dict(epoch=epoch, learning_rate=lr, training_loss=loss,
                                    validation_accuracy=accuracy, training_seconds=total_training_seconds,
                                    epoch_training_seconds=seconds,
                                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))
                payload = dict(schema=SCHEMA_VERSION, identity=identity, epoch=epoch,
                               model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                               random_states=random_states(), history=history,
                               scheduler=dict(last_epoch=epoch, full_epochs=protocol.full_epochs, warmup_epochs=protocol.warmup_epochs))
                save_checkpoint(checkpoint_path, payload)
                write_json(directory / "history.json", history)
                print(f"epoch {epoch}/{protocol.full_epochs}: val={accuracy:.4f}, train={seconds:.1f}s", flush=True)
            if pause_check is not None and pause_check():
                requested[0] = True
            completed = history[-1]["epoch"] if history else 0
            status = "complete" if completed == protocol.full_epochs else "target_reached" if completed == target_epoch else "paused"
            result = dict(status=status, epoch=completed, candidate_id=candidate.id,
                          parameters=parameters, full_epochs=protocol.full_epochs,
                          validation_accuracy=history[-1]["validation_accuracy"] if history else None,
                          training_seconds=total_training_seconds, identity=identity,
                          accounting_uncertain=uncertain, pause_requested=requested[0])
            # The checkpoint is authoritative if interruption occurred before the JSON mirror write.
            write_json(directory / "history.json", history)
            write_json(directory / "result.json", result)
            return result
        except Exception as exc:
            session["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            session.update(status=status, active_seconds=time.perf_counter() - started)
            session["overrun_seconds"] = max(0.0, session["active_seconds"] - session_seconds)
            write_json(session_log_path, sessions)

