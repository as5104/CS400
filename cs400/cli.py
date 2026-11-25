"""Manual commands; help, preflight and plans never train or download."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .config import Candidate, Protocol, DEPTH_NAMES
from .storage import read_json


def emit(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def parser():
    root = argparse.ArgumentParser(description="CS400 core: training starts only with an explicit manual command")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Inspect dependencies/device and count model parameters; no training")
    plan = sub.add_parser("plan", help="Show configured search schedule without running it")
    plan.add_argument("--protocol", default="configs/pilot.json")
    prepare = sub.add_parser("prepare-cifar10", help="Prepare saved complementary splits; no training")
    prepare.add_argument("--data-root", default="data")
    prepare.add_argument("--manifest", default="data/cifar10-split.json")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--download", action="store_true", help="Explicitly allow downloading CIFAR-10")
    folder = sub.add_parser("prepare-folder", help="Validate explicit train/validation/test class folders; no training")
    folder.add_argument("--data-root", required=True)
    folder.add_argument("--manifest", required=True)
    folder.add_argument("--confirm-augmentation-safe", action="store_true")
    train = sub.add_parser("train", help="Manually start/resume one candidate")
    train.add_argument("--candidate", default="configs/reference.json")
    train.add_argument("--protocol", default="configs/pilot.json")
    train.add_argument("--data-root", default="data")
    train.add_argument("--manifest", default="data/cifar10-split.json")
    train.add_argument("--run-dir", required=True)
    train.add_argument("--target-epoch", type=int, required=True)
    train.add_argument("--session-minutes", type=float, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--start-training", action="store_true", help="Required acknowledgement that this command trains")
    search = sub.add_parser("search", help="Manually start/resume a calibrated synchronous search")
    search.add_argument("--protocol", required=True, help="Frozen pilot-confirmed protocol with an active-time budget")
    search.add_argument("--data-root", default="data")
    search.add_argument("--manifest", default="data/cifar10-split.json")
    search.add_argument("--run-dir", required=True)
    search.add_argument("--session-minutes", type=float, required=True)
    search.add_argument("--epoch-cost-seconds", type=float, required=True, help="Conservative pilot estimate including validation/checkpoint overhead")
    search.add_argument("--method", choices=("motpe", "random"), default="motpe")
    search.add_argument("--no-pruning", action="store_true")
    search.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    search.add_argument("--resume", action="store_true")
    search.add_argument("--start-training", action="store_true")
    status = sub.add_parser("status", help="Read saved run metadata and metrics")
    status.add_argument("--run-dir", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate a full-budget model on clean test and CIFAR-10-C")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--manifest", default="data/cifar10-split.json")
    evaluate.add_argument("--data-root", default="data")
    evaluate.add_argument("--corruption-root", help="Required for CIFAR-10-C evaluation")
    evaluate.add_argument("--clean-only", action="store_true", help="Evaluate only held-out clean test images")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    export = sub.add_parser("export", help="Export a completed model with class and preprocessing metadata")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--manifest", default="data/cifar10-split.json")
    export.add_argument("--output", required=True)
    predict = sub.add_parser("predict", help="Classify an image using a portable model export")
    predict.add_argument("--model", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    analysis = sub.add_parser("analyze", help="Optional post-search regression; needs completed evaluation reports")
    analysis.add_argument("--reports", nargs="+", required=True)
    analysis.add_argument("--output", required=True)
    comparison = sub.add_parser("compare", help="Compare frozen searches using validation objectives; no training")
    comparison.add_argument("--run-dirs", nargs="+", required=True)
    comparison.add_argument("--output-dir", required=True)
    comparison.add_argument("--parameter-reference", type=int, default=6_000_000)
    comparison.add_argument("--seconds-reference", type=float, default=3600.0)
    return root


def execute(args):
    if args.command in ("train", "search") and not args.start_training:
        raise ValueError("Training was not started. Add --start-training when you are ready")
    if args.command == "preflight":
        import importlib.metadata
        import torch
        from .model import ResNet, parameter_count
        architectures = []
        for depth in DEPTH_NAMES:
            for width in (16, 24, 32):
                model = ResNet(Candidate(depth=depth, width=width))
                architectures.append(dict(depth=depth, width=width, parameters=parameter_count(model)))
        emit(dict(training_started=False, dataset_downloaded=False, python=sys.executable,
                  versions={m: importlib.metadata.version(m) for m in ("torch", "torchvision", "numpy", "optuna")},
                  cuda_available=torch.cuda.is_available(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                  architectures=architectures, note="Parameter counts do not prove training memory fit or speed"))
    elif args.command == "plan":
        protocol = Protocol(**read_json(args.protocol))
        counts = [protocol.cohort_size, *protocol.survivors]
        previous = 0
        model_epochs = 0
        for epoch, count in zip(protocol.rungs, counts):
            model_epochs += (epoch - previous) * count
            previous = epoch
        emit(dict(protocol=asdict(protocol), protocol_id=protocol.id,
                  startup_model_epochs=protocol.startup_trials * protocol.full_epochs,
                  cohort_model_epochs=model_epochs, training_started=False,
                  main_search_ready=protocol.pilot_confirmed and protocol.active_search_seconds is not None))
    elif args.command == "prepare-cifar10":
        from .data import prepare_cifar10
        value = prepare_cifar10(args.data_root, args.manifest, args.seed, args.download)
        emit(dict(manifest=str(Path(args.manifest).resolve()), id=value["id"],
                  training=len(value["train"]), validation=len(value["validation"]), training_started=False))
    elif args.command == "prepare-folder":
        from .folder_data import prepare_folder
        value = prepare_folder(args.data_root, args.manifest, args.confirm_augmentation_safe)
        emit(dict(manifest=str(Path(args.manifest).resolve()), id=value["id"], classes=value["classes"],
                  counts={split: len(value[split]) for split in ("train", "validation", "test")}, training_started=False))
    elif args.command == "predict":
        from .inference import predict_image
        emit(predict_image(args.model, args.image, args.device))
    elif args.command == "train":
        from .trainer import train_candidate
        emit(train_candidate(Candidate(**read_json(args.candidate)), Protocol(**read_json(args.protocol)),
                             args.data_root, args.manifest, args.run_dir, target_epoch=args.target_epoch,
                             session_seconds=args.session_minutes * 60, device_name=args.device, resume=args.resume))
    elif args.command == "status":
        directory = Path(args.run_dir)
        emit({name: read_json(directory / f"{name}.json") if (directory / f"{name}.json").exists() else None
              for name in ("run", "result", "sessions")})
    elif args.command == "search":
        from .search import search_session
        report = search_session(Protocol(**read_json(args.protocol)), args.data_root, args.manifest, args.run_dir,
                                session_seconds=args.session_minutes * 60, estimated_epoch_seconds=args.epoch_cost_seconds,
                                method=args.method, pruning=not args.no_pruning, device_name=args.device, resume=args.resume)
        emit(dict(frozen=report["frozen"], stop_reason=report["stop_reason"], active_seconds=report["active_seconds"],
                  completed=len(report["completed"]), pareto_count=len(report["pareto"])))
    elif args.command == "evaluate":
        from .evaluation import evaluate_run, evaluate_clean
        if args.clean_only:
            if args.corruption_root:
                raise ValueError("Choose clean-only or corruption evaluation")
            report = evaluate_clean(args.run_dir, args.manifest, args.data_root, args.output, args.device)
        else:
            if not args.corruption_root:
                raise ValueError("Supply --corruption-root, or explicitly select --clean-only")
            report = evaluate_run(args.run_dir, args.manifest, args.data_root, args.corruption_root, args.output, args.device)
        emit(report["metrics"])
    elif args.command == "export":
        from .evaluation import export_run
        export_run(args.run_dir, args.manifest, args.output)
        emit(dict(export=str(Path(args.output).resolve())))
    elif args.command == "analyze":
        from .analysis import analyze_reports
        emit(analyze_reports(args.reports, args.output))
    elif args.command == "compare":
        from .comparison import compare_searches
        report = compare_searches(args.run_dirs, args.output_dir, args.parameter_reference, args.seconds_reference)
        emit(dict(output=str(Path(args.output_dir).resolve()), searches=len(report["searches"]), training_started=False))


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        execute(args)
    except (ValueError, OSError, RuntimeError) as exc:
        command_parser.exit(2, f"Error: {exc}\n")

