# CS400 experiment settings

Finalized for the available 16–24 GPU-hour allowance, 13 November 2026.
This document specifies the main experiment. The nine-epoch integration test
and the earlier standalone pilots remain separate evidence of implementation
correctness; they are not additional main-search trials.

## Scope and interpretation

Run the four-method comparison on CIFAR-10, on the same RTX 4060 laptop.
Use one recorded seed (42). This is an exploratory, resource-limited comparison,
not a statistically conclusive benchmark. Do not claim that MOTPE or pruning
is superior unless the observed measurements support the specific descriptive
claim; do not infer statistical significance from one search per method.

The implemented dataset adapter is CIFAR-10. Arbitrary uploaded datasets and
their corruption tests are not implemented and must not be described as working.
The core works locally. The presentation/dashboard layer remains separate work.

## Frozen training and search settings

Use `configs/experiment-seed42.json` for every arm:

| Setting | Value |
|---|---|
| Train / validation / held-out test | Saved 45,000 / 5,000 / 10,000 split |
| Full candidate budget | 60 epochs |
| Startup | 4 fully trained candidates, reference plus 3 sampled choices |
| Later cohort size | 4 distinct candidates |
| Pruning schedule | 4 at 15 epochs, 2 at 30, 1 at 60 |
| Without pruning | All 4 reach 60 epochs |
| Batch / precision / workers | 128 / FP32 / 0 |
| Learning-rate schedule | 5-epoch warmup, then cosine to 1% at epoch 60 |
| Search seed | 42 |
| Active-time cap per arm | 18,900 seconds = 5 hours 15 minutes |
| Admission estimate | 39 seconds per model-epoch, including overhead |
| Default manual session | 30 minutes; change session length if convenient |

All architecture, optimizer and regularization ranges, promotion objectives,
crowding-distance tie resolution, and final Pareto objectives remain as in the
approved framework. The numerical changes are startup 12 -> 4 and cohort
12 -> 6 -> 3 becoming 4 -> 2 -> 1. These reduce cost while preserving full
60-epoch training and both intermediate rung positions. Four startup observations
provide a small initial MOTPE history, so proposal quality remains uncertain.

The three full-budget pilots support memory fit and sustained training at these
settings. They do not establish that the pruning rule reliably retains all useful
models across the whole search space. Report that limitation and use the no-pruning
arms to provide counter-evidence where available. Keep test/corruption accuracy out
of protocol selection and optimization.

## Budget and feasibility

Pilot active time averaged 27.54–31.27 seconds per epoch. The largest observed
training-only epoch was 37.24 seconds. The 39-second admission estimate includes
allowance above the measured averages for validation, checkpointing and variability;
it is a planning estimate, not a worst-case guarantee. All arms use the same value.

Startup costs 240 model-epochs. A pruned cohort costs 120 additional model-epochs;
a full-training cohort costs 240. At 39 seconds/epoch, startup plus one cohort is
3.9 hours with pruning and 5.2 hours without pruning. Both fit the 5.25-hour cap.
Actual pilot averages suggest roughly two hours for startup. The number of later
cohorts depends on actual speed and admission checks; do not promise a fixed count.

Four caps total 21 active search hours. Plan approximately three additional hours
for post-search evaluation and contingency. A throughput check on the largest
pilot took 3.45 seconds for 10,000 corrupted examples, projecting approximately
4.4 minutes for 75 corruptions/severities plus clean evaluation per such model.
This is a projection, not an end-to-end evaluation-time guarantee. Record actual
evaluation time and review the remaining allowance before launching a large batch.
If the allowance is exhausted, report unfinished evaluation rather than selecting
only favorable models or silently claiming complete coverage.

Intentional pauses are excluded from active search time. Startup, failed attempts,
validation, checkpointing and proposal overhead are charged. An epoch may overrun
a limit; report the overrun, and exclude candidates completed after the global cap.
The controller may stop early when another cohort cannot fit its estimate. Compare
both the common allocated cap and actual used time; do not describe unequal actual
time as identical compute. Unused budget is part of the measured system behavior.

## Four comparison arms

| Run directory | Proposal method | Pruning |
|---|---|---|
| runs/main-random-full | Random | No |
| runs/main-motpe-halving | MOTPE | Yes |
| runs/main-random-halving | Random | Yes |
| runs/main-motpe-full | MOTPE | No |

This fixed execution order is declared before main results are observed. Keep the
laptop plugged in with the same power mode and no other GPU workload. Record
conditions that noticeably affect timings; machine and thermal variability remain
limitations. Each arm trains and pays for its own identical startup configurations.
Do not import pilot checkpoints or another arm's measured results for free.

Use this command for the first arm, manually:

```powershell
.\.venv\Scripts\python.exe -m cs400 search --protocol configs/experiment-seed42.json --run-dir runs/main-random-full --method random --no-pruning --session-minutes 30 --epoch-cost-seconds 39 --start-training
```

For a normal pause, repeat exactly with `--resume`. For the other arms use their
directory and method from the table; supply `--no-pruning` only for the full-training
arms. Never run two training sessions concurrently. Unexpected process termination
requires accounting review; the current controller intentionally refuses to hide
unknown costs. Wait for a normal session exit before shutting down.

Freeze code before starting an arm. Training identities include a source fingerprint;
editing `cs400/*.py` prevents normal resumption. Existing completed models remain
evaluable/exportable. Put later presentation-only work outside that package while
searches are active. A necessary training-code fix requires explicit experiment
versioning rather than overriding stored identities.

## Comparison measures

Report per arm: eligible completed models, observed Pareto count, best final-epoch
validation accuracy, parameters, measured training seconds, active time, cap usage,
overrun, and stopping reason. Include startup in all counts and costs, and identify
it separately when interpreting guided-search progress.

Use exact three-objective hypervolume with minimization coordinates
`(1 - validation_accuracy, parameters / 6,000,000, training_seconds / 3,600)`
and common reference `(1, 1, 1)`. Higher volume is better under these scales.
The reference is fixed before main results. If a model exceeds either scale,
the comparison tool refuses to clip it silently: document the problem and use a
larger common reference for every arm in an explicitly labeled revised analysis.
Also report the raw metrics, which do not depend on this reference choice.

Compute progress curves at archived completion times using only validation
objectives. Never include corruption or clean test accuracy in hypervolume or
search-time Pareto selection. The comparison command verifies matching protocols,
datasets, hardware/code identity and admission estimate, with only method, pruning
and seed allowed to differ.

```powershell
.\.venv\Scripts\python.exe -m cs400 compare --run-dirs runs/main-random-full runs/main-motpe-halving runs/main-random-halving runs/main-motpe-full --output-dir output/main-comparison
```

## Post-search evaluation and final report

After each archive is frozen, evaluate every eligible completed checkpoint on the
clean test set and the standard 15 CIFAR-10-C corruptions at five severities. Keep
these results out of subsequent search choices. Report average corruption error
(not reference-normalized mCE), corruption accuracy, and clean-minus-corrupted
accuracy in percentage points. Evaluate repeated trained models separately, even
if their configuration IDs match across arms; identities include protocol/code.

Use descriptive robustness comparisons by default. Fit a configuration-to-robustness
predictor only if enough distinct configurations and useful held-out prediction
support it; the existing implementation requires at least 40 distinct configurations
before attempting fitting. That is a practical guard, not a theorem about sample
size. Do not extend training merely to force a feature-importance plot.

Retain configuration files, source snapshot, split manifest, hardware/version
metadata, SQLite ledgers, epoch histories, checkpoints, all condition-level
evaluation reports and comparison exports. The final project report should separate
implementation validation, pilot results, and main experimental results, and state
the single-seed, limited-initial-history, timing, pruning and dataset-scope limitations.
No main result or success claim is to be filled in before its experiment completes.
