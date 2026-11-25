# Local Multiobjective Search for ResNet Configurations

CS400 academic project by **Ankit Sarkar (2301033)**, supervised by **Dr. Parashjyoti Borah**, Indian Institute of Information Technology Guwahati.

A local image-classification system that jointly searches ResNet architecture, optimizer and regularization settings. It returns an observed Pareto set balancing **validation accuracy, parameter count and measured training time**. The dashboard supports dataset uploads, explicit training controls, pause/resume, result comparison, model export and image prediction.

## Repository branches

- **`main`**: the completed search framework, tests, documentation and selected experiment evidence.
- **[`cnn-resnet-study`](https://github.com/as5104/CS400/tree/cnn-resnet-study)**: the earlier CNN/ResNet learning implementation and comparisons, preserved at the previous `main` commit (`bf46189`).

## Setup

Use Python 3.11 or newer and a PyTorch/torchvision build suitable for your machine. The recorded experiments used Python 3.13, PyTorch 2.9.1+cu130, torchvision 0.24.1+cu130, NumPy 2.2.6 and Optuna 4.5.0 on an RTX 4060 Laptop GPU with 8 GB VRAM.

From a fresh clone in PowerShell:

```powershell
git clone https://github.com/as5104/CS400.git
cd CS400
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install an appropriate **torch/torchvision pair** using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the project:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m cs400 preflight
```

On Linux/macOS, use `.venv/bin/python` in place of the Windows executable. CUDA requires a compatible NVIDIA GPU and PyTorch build; CPU operation is available but slower. Preflight inspects the environment and counts model parameters without training or downloading data.

## Run the dashboard

```powershell
.\.venv\Scripts\python.exe tools/build_dashboard_demo.py
.\.venv\Scripts\python.exe -m cs400.dashboard
```

Open **http://127.0.0.1:8765**. The first command generates the optional 400-image synthetic example at `output/demo/synthetic-shapes-demo.zip`; it does not train a model.

1. Upload a supported PNG/JPEG classification ZIP, or use the generated example.
2. Review classes, splits, transformations and search settings.
3. Click **Start local search** when ready to train.
4. Use **Pause** and wait for confirmation before closing the server or shutting down.
5. After completion, inspect the Pareto models, clean-test results and downloadable exports.

Opening the dashboard or uploading a dataset does not start training. Keep the server terminal open during a job. See [the local workflow guide](docs/LOCAL_DEMO.md) for accepted folder layouts, limits, calibration and recovery behaviour.

## How the search works

1. Validate the dataset and keep training, validation and test splits separate.
2. Propose configurations over 15 ResNet depth/width combinations, SGD or AdamW, learning rate, weight decay, dropout and augmentation.
3. Use Optuna's multiobjective TPE sampler (MOTPE) with completed-trial history.
4. After four fully trained startup configurations, run synchronous cohorts. The main protocol trains four candidates to 15 epochs, two to 30, and one to 60. Promoted candidates resume their checkpoints.
5. Rank candidates using non-dominated sorting and crowding distance. Intermediate promotion uses validation performance, parameter count and estimated full-budget training time.
6. Select the final Pareto set from eligible completed models using validation accuracy, parameter count and actual training time. Evaluate held-out clean data and supported corruption benchmarks afterward.

The Pareto set is optimal among the eligible configurations observed in that run, not a claim of global optimality. Test results never guide proposals, promotion or search-time Pareto selection.

## Main experiments and evidence

Four CIFAR-10 searches compared random and MOTPE proposals, with full or staged training. They produced **29 completed checkpoints representing 15 distinct configurations**, followed by clean-test and 75-condition CIFAR-10-C evaluation for each checkpoint.

| Search setting | Completed | Pareto | Active search hours |
| --- | ---: | ---: | ---: |
| Random, full training | 8 | 5 | 3.88 |
| MOTPE, staged training | 7 | 5 | 4.88 |
| Random, staged training | 6 | 4 | 4.06 |
| MOTPE, full training | 8 | 4 | 4.49 |

The shared reference configuration achieved **92.12% validation**, **91.47% clean-test** and **71.584% mean corruption accuracy**. All arms used one seed and the same 5.25-hour allocation, but consumed different active times. The MOTPE/full run contains a documented battery-power interval. These experiments demonstrate the framework and recorded trade-offs; they do not establish general search-method superiority.

The separate synthetic dashboard demonstration completed 38 models and produced three Pareto exports, each scoring 60/60 on its clean test set. It verifies the workflow on a simple task, not real-world generalization.

- [Experiment protocol and commands](docs/CS400_Experiment_Protocol.md)
- [Results and interpretation](output/final-results/Results_Summary.md)
- [Per-model measurements](output/final-results/models.csv), [corruption conditions](output/final-results/corruption-conditions.csv) and [epoch histories](output/final-results/epoch-history.csv)
- [Saved custom-demo audit](output/final-verification/custom-dashboard-demo.json)
- [Final report PDF](output/pdf/CS400_Final_Project_Report.pdf) and [editable LaTeX source](output/latex/CS400_Final_Report/main.tex)
- [Concise framework](docs/CS400_System_Framework_Concise.md)

The earlier framework is a planning document. The executed cohort schedule and results are specified in the experiment protocol and final report.

## Tests and optional analysis

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite uses simulated training, synthetic data and small CPU operations. It does not start a real search or download datasets.

For report figure generation or the optional analysis module:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[reporting,analysis]"
```

See [the reporting tools guide](tools/README.md). `evaluate_main.py` and `build_results.py` work with locally retained frozen runs; the published CSV/JSON evidence can be read without those checkpoints. No robustness predictor or feature-importance claim was fitted to the 15-configuration main sample. Custom corruption benchmarks and SHAP are not implemented.

## What is kept local

`.gitignore` excludes datasets, uploaded files, virtual environments, run databases, checkpoints, model exports, worker logs, caches, temporary builds, demo ZIPs and duplicate document exports. Only explicitly selected experiment evidence and the final report are allowed under `output/`.

A fresh clone has no trained models or saved jobs. The dashboard can show the published reference summary and CSVs, but historical per-model inspection/export requires the original local `runs/` and dataset manifests. Generate the demo ZIP or upload your own dataset to create new jobs.

Historical source snapshots and full audit archives remain in the original local workspace and are not bundled in Git. Reports may refer to these local artifacts. New source versions change the training identity: use new run directories for new experiments and retain the original source with historical checkpoints when recovery is needed.

## Project structure

```text
cs400/          Models, data handling, search, evaluation and local dashboard
configs/        Reference, pilot, smoke and main experiment settings
tests/          Core, search, evaluation and local workflow tests
docs/           Framework, experiment protocol, report text and usage guide
tools/          Synthetic demo and report-generation scripts
output/         Explicitly selected results and final report only
```
