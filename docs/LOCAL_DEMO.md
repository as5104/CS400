# Local dataset-to-model workflow

## Start the application

For a fresh Git clone, first follow the environment setup in [README.md](../README.md).
Generate the optional upload example with
`.\.venv\Scripts\python.exe tools/build_dashboard_demo.py`. The demo ZIP, saved jobs,
datasets and model weights are local files and are not included in Git.

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m cs400.dashboard
```

Open http://127.0.0.1:8765. The application uses the Python environment running this command. Other computers need Python, the project dependencies, and a suitable PyTorch installation. CUDA is offered only when PyTorch detects it; CPU is available but slower. Data, checkpoints and computation stay on the local computer.

Opening the page and uploading a dataset do not start training. Only **Start local search** or **Resume** starts the worker. Keep the terminal open. Closing the browser does not stop training.

## Upload and run

1. Upload a ZIP containing PNG/JPEG classification images in one of the formats below. Confirm that the labels, split independence and training transformations are appropriate.
2. Review the validated class names and training/validation/test counts. You can reuse an uploaded dataset for another independent search.
3. Keep the standard settings or choose a short demonstration. Optional settings include epoch budget, total active search hours, session length, batch size, device and seed. The architecture, optimizer and regularization choices are searched automatically within the supported search space.
4. Press **Start local search** yourself. A separate local worker first trains the largest supported architecture with AdamW for three calibration epochs. This checks the selected batch size and estimates cost, with a 25% allowance. It does not guarantee sustained speed or rule out later failures.
5. If the estimated startup plus one cohort exceeds your budget, the job waits. Increase the budget on the page before search starts, or create a new job with fewer epochs. Budget changes are blocked once the search database exists.
6. Search trains four startup models, then MOTPE proposes cohorts of four candidates. For a full budget B, all four train to B/4 epochs, two continue to B/2, and one continues to B. Promotion uses non-dominated sorting and crowding distance. The loop repeats while the active budget admits another cohort.
7. Use **Pause** when needed. Wait until the status says paused before shutting down. The current epoch finishes and its checkpoint is saved. **Resume** continues the same configuration and learning-rate schedule. The session limit also pauses at safe boundaries; it may be exceeded by the current epoch or evaluation.
8. When search ends, the worker evaluates eligible completed models on the held-out clean test set and exports the observed Pareto set. If the session ends during evaluation, resume to finish evaluation and exports without repeating training.
9. Choose **View Pareto models**. Compare validation accuracy, parameter count and actual training time; inspect the clean-test score and learning curve. Select a model, export/download its weights, or try a domain-appropriate image with CPU inference. Download the run's results CSV.

Calibration and final evaluation are additional to the active search budget. The display's active time is the last saved controller value, not a live stopwatch. One dashboard-managed compute job runs at a time. Avoid launching separate CLI training simultaneously. Keep the laptop plugged in and avoid other heavy work when comparing times.

A forced process kill or power loss can make active-time accounting uncertain. The application retains files and reports the interruption instead of silently resuming a controlled experiment with guessed costs. Normal Pause/Resume is the supported recovery path. Stop a running job through the page before stopping the server; Ctrl+C on the server also requests a worker pause.

## Accepted ZIP contents

Class folders with at least ten distinct images per class:

```text
dataset.zip
  class_a/image1.png
  class_a/image2.jpg
  class_b/image1.png
  ...
```

The importer saves a per-class split of approximately 70% training, 15% validation and 15% test using seed 42. The search seed does not change this already saved split. Automatic splitting rejects identical decoded images. It cannot detect all near-duplicates or related subjects.

For subject/source-based independence, provide your own splits:

```text
dataset.zip
  train/class_a/...
  train/class_b/...
  validation/class_a/...
  validation/class_b/...
  test/class_a/...
  test/class_b/...
```

All three splits must have the same classes and at least one image per class. Identical decoded images crossing splits or labels are rejected. One outer dataset directory is allowed. Keep images directly inside class folders; remove README files and operating-system metadata from the ZIP.

Limits: 512 MiB archive, 2 GiB expanded data, 100,000 images, 2–100 classes, 20 MiB and 16 megapixels per image. ZIP links, unsafe paths, encrypted entries and unsupported files are rejected. Accepted files are copied into `data/uploads/<dataset-id>` with a manifest. Files must remain unchanged during a run.

Preprocessing is RGB, bilinear resize to 32 by 32 and fixed per-channel mean/std 0.5. Training uses random crop and horizontal flip, with erasing as a searched choice. Use this workflow only when these transformations preserve labels. It is not a general solution for fine-detail or orientation-sensitive domains.

## Short manual demonstration

Use `output/demo/synthetic-shapes-demo.zip` (also linked on the page). It contains 400 generated images of circles and squares: 280 training, 60 validation and 60 test. Select **Short workflow demonstration**: 12 full epochs, 0.25 active search hours and a 15-minute session. Calibration may request more budget on slower systems. This is a functional example, not a realistic benchmark or proof of better search quality. It contains no trained weights.

The new upload and job workflow has automated tests with simulated training and browser checks. The user has now completed a real CUDA run on the synthetic shapes dataset: 140 tried configurations, 38 completed models and three Pareto exports in 834.80 active search seconds. All 68 promotion decisions and 38 checkpoint/evaluation identities were audited; all three exports reproduced 60/60 clean test predictions on CPU. See `output/final-verification/custom-dashboard-demo.json`. This is a functional demonstration, not a realistic custom-domain benchmark. The four completed academic searches are independent CIFAR-10 evidence.

## Results and saved evidence

Jobs and logs: `workspace/jobs/<job-id>`. Uploaded datasets: `data/uploads/<dataset-id>`. Search records: `runs/local-<job-id>`, including SQLite history, candidate checkpoints, epoch records, evaluations and portable exports. The final Pareto set uses full-budget validation accuracy, parameter count and measured training time. Test accuracy never guides proposals, promotion or Pareto membership. The set is optimal among eligible observed candidates, not a guarantee of a global optimum.

Custom corruption evaluation is not implemented because it needs an appropriate domain benchmark. Custom jobs report clean test accuracy; CIFAR-10-C is rejected for custom classes. Portable exports include weights, architecture, classes and preprocessing, without optimizer state. Softmax scores are not calibrated confidence.

In the original research workspace, the **Models** and **Reference experiment** sections retain all four historical searches and 29 clean/CIFAR-10-C evaluations. Select `main-motpe-halving` to show seven completed models and five observed Pareto models. A fresh clone includes the reference summary and selected CSV evidence in `output/final-results`, but has no historical jobs or model weights. Its Models list fills as you run new searches. Generated plots remain local and can be rebuilt from the published data.

The original implementation is retained in `output/experiment-source-seed42.zip`, with settings in `output/experiment-freeze-seed42.json`. New dashboard functionality was added after those experiments. Historical source fingerprints prevent resuming them silently with changed code. Do not edit frozen times or measurements. Preserve `runs`, manifests, uploaded data, `workspace/jobs`, protocols, exports and source snapshots together.
