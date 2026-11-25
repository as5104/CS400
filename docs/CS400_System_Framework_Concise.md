# CS400: Final System Framework

Automated Multi-Objective Search of ResNet Configurations

Approved method. Numerical training and budget settings are confirmed by pilot runs before the main experiments. Finalized 30 October 2026.

## 1. Input and dataset preparation

The user selects a supported, labelled image-classification dataset and search budget. Check files, labels, and split overlap. For CIFAR-10, save a stratified split of 45,000 training and 5,000 validation images; keep the official 10,000 test images separate. Correct and verify the existing split logic before experiments.

Training data updates weights; validation guides search. Resize only when required and use fixed normalization, with any fitted statistics obtained from training data only. Random augmentation applies only to training. Custom datasets need a declared input format, valid splits, and label-preserving transformations.

## 2. Joint search space and candidate generation

One candidate combines architecture, optimizer, and regularization choices from this initial space.

| Search part | Initial choices |
|---|---|
| Architecture | Basic ResNet blocks; stage depths (1,1,1,1), (1,2,2,1), (2,2,2,2), (2,3,3,2), or (3,3,3,3). Base width 16, 24, or 32; double at each later stage. |
| Optimizer | SGD (momentum 0.9; learning rate 0.01-0.1) or AdamW (learning rate 0.0001-0.003). Sample learning rates logarithmically. |
| Regularization | Weight decay 0.000001-0.01 (logarithmic); dropout 0, 0.1, or 0.2 after pooling; basic crop/flip augmentation, with or without random erasing, for CIFAR-10. |

Configure Optuna's TPESampler for MOTPE (multi-objective Tree-structured Parzen Estimator) [1]. Initially train 12 startup configurations to full budget without pruning: one fixed reference and 11 sampled choices, covering both optimizers. Then propose cohorts from completed results. Partial results guide only the separate pruning controller.

## 3. Train and measure the candidate

Use PyTorch, cross-entropy loss, and one candidate at a time on the local GPU. Fix batch size and precision for the study. Use five-epoch linear warmup followed by cosine decay to 1% of the chosen learning rate, defined for the entire training budget.

Record validation accuracy each epoch, parameter count, and training time. Save weights, optimizer, scheduler, random states, and accumulated time so promotion or pause/resume continues the same training process without restarting the learning-rate schedule.

## 4. Multi-fidelity training and pruning

Pilot schedule: full budget B = 60 epochs. After startup, each cohort follows 12 candidates at 15 epochs -> 6 at 30 -> 3 at 60. Epochs are cumulative; survivors resume checkpoints. Wait until all valid candidates reach the same rung before comparison.

At each promotion rung, maximize mean validation accuracy over the last three epochs; minimize parameter count and estimated full-budget training time = time so far x B / epochs completed.

Fill slots by non-dominated rank: the first rank contains non-dominated candidates; remove them to find subsequent ranks. If a rank exceeds the remaining slots, prefer larger crowding distance [2]: sum neighboring gaps divided by each objective's range, with priority for boundary points. Constant objectives contribute zero. This favors different trade-offs. Break ties by higher accuracy, lower estimated time, then configuration ID.

Stop unselected candidates. If fewer valid candidates remain than slots, continue all valid ones; if none remain, end the cohort. Failed or stopped trials never receive invented completed metrics.

## 5. Repeat the optimization loop

Archive every valid candidate that completes B epochs, including startup trials. After each cohort, feed completed results into MOTPE and propose the next cohort. Keep stopped and failed trials separate; save cohort membership, decisions, seeds, and search state for recovery.

The overall budget is total active search time: startup, all attempted training, validation, checkpointing, proposals, and failure overhead. Exclude intentional pauses; budget post-search evaluation separately. Start or extend a cohort only when its estimated cost fits. At the limit, save at the next epoch boundary and record any overrun; only results completed by the cutoff qualify. A separate session limit allows work across laptop sessions.

For the model's training-time objective, include data loading and training operations, synchronizing the GPU at timing boundaries. Exclude validation, checkpoint writes, and pauses; sum across resumed sessions. Record and keep hardware and measurement settings consistent. This cost is machine-specific.

## 6. Select the Pareto-optimal configurations

Compare eligible completed models under the same B-epoch protocol: maximize final-epoch validation accuracy, minimize parameter count (a model-size proxy), and minimize actual accumulated training time. Do not substitute the best earlier checkpoint or estimated time.

A model is dominated if another is at least as good on all three objectives and strictly better on at least one. Keep all non-dominated models, with no fixed output count: the observed Pareto set among eligible completed configurations, not a guaranteed global optimum.

## 7. Evaluate and characterize corruption robustness

Freeze the search results, then evaluate all eligible completed models on held-out clean test data and corrupted images. For CIFAR-10, use the 15 standard CIFAR-10-C corruptions at five severities [3]. Custom datasets need a separately defined corruption test.

Equally average classification error over the 75 corruption/severity pairs. Also report clean-test accuracy minus average corrupted accuracy, in percentage points. Use configuration features in random forest regression; interpret importance or SHAP only with enough distinct models and useful prediction on held-out configurations. Otherwise report descriptive results. Associations do not establish causes.

Neither clean test nor corruption results influence MOTPE, promotion, or search-time Pareto selection. Robustness is post-search analysis, not a fourth search objective.

## 8. Present the final results

The local dashboard combines the observed Pareto set, validation and clean-test accuracy, parameter counts, measured training times, and robustness results. Users compare trade-offs and export selected model weights, configuration, class labels, and preprocessing settings. The complete workflow runs on the user's own computer.

## Implementation and validation conditions

Pilot confirmation: test memory, sustained training speed, early-versus-final rankings, and corruption-evaluation cost on the RTX 4060 (8 GB). Begin with batch size 128 and test a common precision mode. Confirm startup count, B, rung positions, and total search hours; allow enough budget for guided cohorts after startup. If pruning loses useful models, retain more candidates, move rungs later, or use no pruning. Freeze the resulting protocol before main comparisons.

Research validation: compare random search and MOTPE, each with and without pruning, using matched active-time budgets and charged startup costs. Compare Pareto quality and resource use; repeat runs with recorded seeds. Set an affordable repetition count during calibration and report variability. Implement and verify the data pipeline, trainer, and recovery before launching the main search.

## Method references

[[1] Optuna 4.5.0: multi-objective TPESampler.](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html)

[[2] Deb et al. (2002): NSGA-II, non-dominated sorting and crowding distance.](https://doi.org/10.1109/4235.996017)

[[3] Hendrycks & Dietterich (2019): Common Corruptions; CIFAR-10-C data.](https://github.com/hendrycks/robustness)

## Complete system flow

The PDF contains the full diagram. Counts and epochs below are initial pilot settings; use the frozen protocol for main runs.

Input and preparation -> Joint search/MOTPE -> Training and measurement.
The first 12 startup configurations train to 60 epochs without pruning and enter the completed archive.
Later cohorts: 12 reach 15 epochs -> rank and promote 6 to 30 -> rank and promote 3 to 60.
Promotion uses non-dominated rank, normalized crowding distance, then a fixed tie rule.
Promoted candidates resume checkpoints; stopped/failed trials stay separate from MOTPE and final Pareto selection.
Full-budget results enter the completed archive and guide the next cohort while active search budget remains.
After search: eligible completed metrics produce the observed Pareto set; all eligible completed models undergo clean-test and corruption evaluation.
Held-out clean test/corruption data flows only to post-search evaluation and never into search decisions.
Both paths join in the local dashboard and model export.
