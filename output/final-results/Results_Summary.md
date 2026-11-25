# Verified main experimental results

All four searches are frozen. All 29 eligible completed checkpoints have clean-test evaluation and 75 corruption/severity results each.

| Search | Tried | Completed | Pareto | Hypervolume | Active hours | Mean clean accuracy | Mean corrupted accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random / full | 8 | 8 | 5 | 0.4692 | 3.88 | 89.66% | 67.97% |
| MOTPE / pruning | 16 | 7 | 5 | 0.4833 | 4.88 | 89.86% | 68.03% |
| Random / pruning | 12 | 6 | 4 | 0.4442 | 4.06 | 90.35% | 68.70% |
| MOTPE / full* | 8 | 8 | 4 | 0.4122 | 4.49 | 89.49% | 67.89% |

## Interpretation

The shared reference model supplies the highest final validation accuracy in every arm (92.12%). It reaches 91.47% clean test accuracy and 71.584% average corrupted accuracy in all four evaluations. Later proposals did not improve its final validation accuracy in this experiment.

MOTPE with pruning has the largest observed hypervolume, using the fixed three-objective reference and actual recorded training times. Actual active-time use differs across arms despite the common 5.25-hour allocation. These single-seed findings describe this experiment.

Mean robustness values describe each method's selected archive. The archives have different configurations and different sizes, so these means alone do not establish a causal advantage for a method or optimizer.

## Measurement limitations

MOTPE without pruning includes a reported battery interval covering reference-model epochs 36–60 and the following candidate's epochs 1–17. Epoch 17 is the reported transition. The saved durations are retained. This affects time objectives, admission decisions and hypervolume comparability. See ../main-motpe-full-power-note.md.

There are 15 distinct configurations among the 29 checkpoints. Per-run measurements are retained because a configuration identifier does not uniquely identify a trained checkpoint. One startup configuration has 91.00% versus 91.01% clean accuracy across runs. A checkpoint comparison found weight differences, with training-metric differences beginning at epoch 51 for the checked pair. The cause is unresolved. The six-epoch reference resume test passed exactly, but does not prove exact reproducibility for every configuration and duration.

Robustness characterization remains descriptive with this limited configuration sample. No random-forest predictor or feature-importance result is fitted. Repeated configurations are not independent random-seed replications.

## Artifacts

models.csv joins configuration, validation, clean-test, corruption and timing measurements by run and configuration. corruption-conditions.csv contains all 2,175 per-condition rows. epoch-history.csv includes completed and pruned candidate histories. method-summary.csv and severity-summary.csv contain the plotted summaries. Plots are available as PNG and PDF. Data fractions in CSV/JSON become percentages only in displayed tables and axes.

Measured active search time totals 17.31 hours. Recorded evaluation time totals 86.88 minutes. These exclude earlier pilots, integration tests and intentional pauses.
