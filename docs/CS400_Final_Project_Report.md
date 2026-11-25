# Local Multiobjective Search for ResNet Configurations

## Final project report

Ankit Sarkar (Roll No. 2301033)

Project Guide: Dr. Parashjyoti Borah, Department of CSE

B.Tech Computer Science and Engineering

Indian Institute of Information Technology Guwahati

CS400 Final Year Project

19 November 2026

### Abstract

This project implements a local framework for jointly searching ResNet architecture, optimizer settings and regularization choices. It returns configurations that balance validation accuracy, parameter count and measured training time. Multi-objective Tree-structured Parzen Estimation (MOTPE) proposes candidates, and a multi-fidelity controller promotes them through equal-budget training stages using non-dominated sorting and crowding distance. Checkpoints support pause/resume, and eligible completed models form the observed Pareto set. Clean-test and corruption evaluation are performed after search decisions are frozen.

Four CIFAR-10 searches compared random and MOTPE proposals, each with and without pruning, on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM. They produced 29 completed checkpoints representing 15 distinct configurations in 17.31 active search hours. All checkpoints were evaluated on the clean test set and 75 CIFAR-10-C conditions. The shared reference achieved 92.12% final validation accuracy, 91.47% clean-test accuracy and 71.584% mean corruption accuracy. MOTPE with pruning produced the largest observed hypervolume, 0.4833, in this single-seed comparison.

The local dashboard was also exercised from dataset upload through training, evaluation, model export and inference. A separate 400-image synthetic shapes demonstration tried 140 configurations, completed 38 models and returned three Pareto models in 834.80 active search seconds. All three exports reproduced 60 correct predictions on the 60 held-out test images during an independent CPU check.

The project delivers an implemented integration of joint configuration search, multi-objective promotion, resumable execution and traceable results. The CIFAR-10 experiment establishes operation and measured trade-offs on the target laptop, while the synthetic demonstration verifies the complete custom-data workflow. These complementary results support the project's central objective: a usable local system that produces several evaluated ResNet configurations for a supported classification dataset.

<!-- pagebreak -->

## 1 Problem and project contribution

### 1.1 Motivation

Choosing an image classifier involves more than selecting the highest observed accuracy. A larger model may require more memory, and a configuration that trains slowly can consume the resources available for further experiments. Architecture, optimizer and regularization choices interact, so tuning each in isolation can miss useful combinations.

This project addresses that practical problem through a bounded joint search. A user supplies a supported classification dataset and an active search budget. The system returns several completed configurations offering different accuracy, size and training-time trade-offs. A local workflow is useful when experiments must be split across laptop sessions and when users need to retain their datasets and models on their own machine.

### 1.2 Objectives

- Implement the approved joint search space and train its candidates consistently.
- Combine completed-history MOTPE proposals with a separately defined multi-objective promotion controller.
- Preserve checkpoints, decisions, configurations and measurements across manual sessions.
- Select a final Pareto set using full-budget validation results and actual training costs.
- Measure clean-test performance and corruption robustness after search decisions are frozen.
- Present results locally and export models with enough metadata for inference.

### 1.3 Relationship to existing work

Residual networks provide the architecture family [1]. Optuna's multi-objective TPESampler supplies the model-based proposal mechanism [2]. Successive resource allocation motivates staged training [3], while non-dominated sorting and crowding distance provide the promotion ordering [4]. CIFAR-10 and CIFAR-10-C provide the experimental dataset and post-search robustness benchmark [5, 6]. AdamW is an existing optimizer with decoupled weight decay [7].

These components are established methods. The project contribution lies in their concrete integration: a completed-only proposal history, an explicit three-objective promotion rule, resumable local execution, consistent measurement boundaries and an auditable results workflow. The custom controller adapts familiar selection principles to equal-epoch training rungs. The original work is the implemented framework and its experimental evaluation; the underlying methods are attributed to their sources.

### 1.4 Scope of the evidence

The evidence consists of four CIFAR-10 searches with seed 42 on one laptop, followed by a separate real GPU dashboard demonstration on a custom synthetic shapes dataset. The supported ResNet family contains 15 depth/width combinations. The completed system includes dataset validation, joint search, clean-test evaluation, model export and inference. Corruption evaluation uses CIFAR-10-C for the CIFAR-10 experiment; custom jobs use their own held-out clean test split.

<!-- pagebreak -->

## 2 Complete system flow

![Figure 1 System flow](../output/final-results/report-figures/system-flow.png)

Figure 1. Implemented main-experiment workflow. Before the repeating cohorts, four common startup configurations train fully to 60 epochs. Training updates weights; validation guides proposals, promotion and Pareto selection. Held-out clean and corrupted images enter only post-search evaluation. Stopped and failed candidates remain in the audit history but do not enter the completed MOTPE archive.

<!-- pagebreak -->

## 3 Dataset preparation and joint search space

### 3.1 CIFAR-10 and split separation

CIFAR-10 contains 50,000 training and 10,000 test images, each 32 by 32 pixels, across ten classes [5]. The implementation saves a stratified 45,000/5,000 training/validation split from the original training set, using seed 42. Each class contributes 4,500 training and 500 validation examples. The official 10,000-image test set remains separate.

The manifest stores indices, class mapping, normalization, image size and a dataset fingerprint. Complementary indices are checked for missing or repeated entries. Training images and labels are fingerprinted again when loaders are constructed. Training uses random augmentation; validation and test use deterministic preprocessing. CIFAR-10 normalization is fixed at means (0.4914, 0.4822, 0.4465) and standard deviations (0.2023, 0.1994, 0.2010).

| Data | Permitted use |
|---|---|
| Training split | Update network parameters |
| Validation split | Propose configurations, promote candidates, select the search-time Pareto set |
| Clean test split | Post-search performance evaluation |
| CIFAR-10-C | Post-search corruption evaluation |

### 3.2 Candidate definition

| Component | Implemented choices |
|---|---|
| Stage block counts | 1-1-1-1; 1-2-2-1; 2-2-2-2; 2-3-3-2; 3-3-3-3 |
| Base width | 16, 24 or 32 channels; double at each later stage |
| Optimizer | SGD with momentum 0.9; or AdamW with betas 0.9/0.999 and epsilon 0.00000001 |
| Learning rate | SGD: 0.01 to 0.1; AdamW: 0.0001 to 0.003; logarithmic sampling |
| Weight decay | 0.000001 to 0.01; logarithmic sampling; excluded from biases and normalization parameters |
| Dropout | 0, 0.1 or 0.2 after global pooling |
| Augmentation | Crop/flip; or crop/flip plus random erasing |

The network uses a 3 by 3 convolutional stem with stride 1, four residual stages, global average pooling, dropout and a classifier. Each basic residual block contains two convolutions; projection shortcuts handle channel or spatial changes. This defines the project's configurable CIFAR-stem ResNet search family. Ten-class parameter counts range from 308,826 to 4,366,250.

The basic training policy uses a 32-pixel random crop with padding 4 and horizontal flip probability 0.5. Erasing, when selected, has probability 0.25, area range 0.02 to 0.20, aspect-ratio range 0.3 to 3.3 and normalized fill value zero. These settings are fixed within the main comparison.

<!-- pagebreak -->

## 4 Training measurement and recovery

### 4.1 Common training procedure

Every eligible final configuration receives 60 cumulative epochs. The main protocol uses cross-entropy loss, batch size 128, FP32 arithmetic and zero data-loader worker processes. Candidates run sequentially on the same GPU. The learning rate increases linearly over five warmup epochs and then follows cosine decay, reaching 1% of its chosen base value at epoch 60. The schedule is defined against the full budget, so promotion does not restart or shorten it.

At the end of every epoch the trainer records training loss, validation accuracy, learning rate, epoch training seconds, accumulated training seconds and peak CUDA allocation. Parameter count is computed from the constructed model. The implementation does not substitute an earlier best validation checkpoint for the final epoch when calculating final search objectives.

### 4.2 Two different time measurements

| Measurement | Included | Excluded |
|---|---|---|
| Model training-time objective | Data loading and transfers inside the epoch loop; forward pass; backward pass; optimizer update | Validation, checkpoint writes, intentional pauses |
| Active search budget | Invocation setup, proposal work, attempted training, validation, checkpointing and failed/stopped work | Time between intentional sessions; post-search evaluation |

CUDA synchronization at training timing boundaries ensures the timer accounts for queued GPU work. Measured time is a cost on the recorded machine under its operating conditions. It is not a hardware-independent compute measure such as FLOPs and does not measure inference latency or energy.

### 4.3 Checkpoint and session behavior

Checkpoints contain model tensors, optimizer state, scaler state, random-generator states, epoch history and schedule metadata. A metadata identity ties the run to its candidate, dataset manifest, protocol, software fingerprint and hardware environment. Writes use temporary files and atomic replacement; the checkpoint is the authority for reconstructing JSON history after an interruption.

A manual session limit requests a stop at an epoch boundary. Ctrl+C also requests a checkpointed pause after the current epoch. A subsequent invocation with --resume continues the saved candidate and schedule. Therefore, a session may run slightly beyond its requested duration. Existing metadata is not silently replaced when settings change.

An unexpected process termination can leave unrecorded active time. The search controller flags uncertain accounting and refuses an unqualified continuation of the controlled experiment. Automatic crash-time reconciliation is not implemented. This is distinct from a normal checkpointed pause, which was exercised repeatedly in the experiment.

### 4.4 Verified session recovery

The six-epoch reference pilot matched its uninterrupted comparison in the checked state. The subsequent pilots and main experiments exercised checkpointed pauses and resumptions over longer training sessions. The experiment audit retains each run's actual measurements and checkpoint identity; the long-run comparison findings are discussed with the experimental interpretation.

<!-- pagebreak -->

## 5 MOTPE and multiobjective promotion

### 5.1 Proposal history

Each comparison begins with four shared startup configurations: the fixed reference and three sampled configurations covering both optimizer families. All startup candidates train to the full budget. Their costs are charged separately in every arm.

For MOTPE, the program explicitly selects Optuna 4.5.0 TPESampler with maximize/minimize/minimize directions for accuracy, parameters and training seconds [2]. Its settings include four startup trials, 24 proposal candidates, multivariate sampling, group decomposition and no constant liar. Optimizer-specific learning-rate parameters represent the conditional search space.

The sampler is reconstructed from eligible completed observations with a recorded proposal seed. Configurations already attempted are excluded. A bounded retry mechanism falls back to a distinct random configuration if necessary. Candidates within a newly proposed cohort use the completed history available at that time. Partial observations are used by the project's controller, not inserted as full-budget Optuna objective values.

### 5.2 Promotion measurements and exact selection rule

The main pruning schedule is four candidates at 15 epochs, two survivors at 30 epochs and one survivor at 60 epochs. These are cumulative budgets. The controller waits until valid candidates reach the same rung before comparing them.

At intermediate rung r, it maximizes mean validation accuracy over the last three epochs and minimizes parameter count and estimated full-budget time. The time estimate is accumulated training seconds multiplied by 60/r. This simple estimate assumes observed training speed is informative about the remaining epochs; changing power or thermal conditions can weaken it.

Promotion fills its fixed number of slots as follows:

1. Find the non-dominated front using the three current-rung objectives. Remove that front and repeat to produce later ranks.
2. Accept complete fronts in rank order while slots remain.
3. If the next front is too large, sort it by descending crowding distance. For each nonconstant objective, sum normalized neighbor gaps; prioritize its boundary points. Constant objectives contribute zero.
4. Resolve remaining ties by higher rung accuracy, then lower estimated time, then stable configuration ID.
5. Resume selected checkpoints to the next budget. Stop unselected candidates and retain their measured histories.

For example, if all four candidates at epoch 15 are non-dominated but only two slots exist, crowding distance supplies the selection order. Non-dominance alone would not resolve the quota. If fewer valid candidates remain than slots, all valid candidates advance; if none remain, the cohort ends.

### 5.3 Admission and final archive

The controller checks whether estimated remaining work fits the active budget before admitting or extending a cohort. At the measured cutoff it saves at an epoch boundary and records any overrun. Only full-budget results completed by the cutoff are eligible. Their final-epoch validation accuracy, parameter count and actual accumulated training time determine the final Pareto set. The intermediate estimates and smoothed rung accuracies are not final objective substitutes.

<!-- pagebreak -->

## 6 Experimental design and resource budget

### 6.1 Comparison arms

| Arm | Proposal method | Allocation | Purpose |
|---|---|---|---|
| Random / full | Random | Every candidate receives 60 epochs | Simple full-training baseline |
| MOTPE / full | Completed-history MOTPE | Every candidate receives 60 epochs | Examine model-based proposals without pruning |
| Random / pruning | Random | 15/30/60-epoch promotion | Examine staged allocation without MOTPE |
| MOTPE / pruning | Completed-history MOTPE | 15/30/60-epoch promotion | Exercise the integrated framework |

This four-arm design evaluates proposal generation and staged allocation separately and together. The comparison reports one run per arm, with shared startup configurations and a common nominal budget.

### 6.2 Frozen main settings

| Setting | Value |
|---|---|
| Device | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |
| Training | 60 epochs; batch 128; FP32; 5-epoch warmup |
| Seed | 42 |
| Startup | Four fully trained shared configurations per arm |
| Later pruned cohorts | 4 at 15 epochs, 2 at 30, 1 at 60 |
| Later full cohorts | Four configurations, each trained for 60 epochs |
| Active allocation per arm | 18,900 seconds, or 5 hours 15 minutes |
| Admission estimate | 39 seconds per model-epoch, including allowance for overhead |
| Software | PyTorch 2.9.1+cu130; torchvision 0.24.1+cu130; NumPy 2.2.6; Optuna 4.5.0 |

Pilot measurements informed the frozen main protocol: four startup trials and 4/2/1 cohorts with cumulative budgets of 15/30/60 epochs. This resource allocation was fixed before the main comparison.

Startup costs 240 model-epochs. A pruned cohort adds 4 x 15 + 2 x 15 + 1 x 30 = 120 model-epochs. A full cohort adds 4 x 60 = 240 model-epochs. The admission estimate is used for planning; actual recorded time determines budget consumption. Unused budget can remain when another cohort does not fit.

### 6.3 Hypervolume definition

Comparison uses minimization coordinates (1 - validation accuracy, parameters/6,000,000, training seconds/3,600), with reference point (1,1,1). Exact three-dimensional hypervolume is the union of boxes between eligible objective points and that reference. Higher volume indicates better coverage at these fixed scales. The implementation rejects points beyond the reference rather than silently clipping them. The same scales apply to all four arms, using each run's recorded training durations.

<!-- pagebreak -->

## 7 Main search results

| Arm | Tried | Completed | Pareto | Active hours | Hypervolume |
|---|---|---|---|---|---|
| Random / full | 8 | 8 | 5 | 3.88 | 0.4692 |
| MOTPE / pruning | 16 | 7 | 5 | 4.88 | 0.4833 |
| Random / pruning | 12 | 6 | 4 | 4.06 | 0.4442 |
| MOTPE / full* | 8 | 8 | 4 | 4.49 | 0.4122 |

Table 1. Saved frozen results, including charged startup trials. Asterisk: the MOTPE/full run contains the reported battery-power interval. All arms ended because another estimated cohort did not fit the remaining allocation.

![Figure 2 Hypervolume progress](../output/final-results/report-figures/hypervolume.png)

Figure 2. Hypervolume changes when eligible full-budget models enter each archive. Curves use actual active time and end at their last eligible completion; shutdown overhead may make final active time slightly greater. Each curve is read against its actual elapsed search time.

MOTPE with pruning tried 16 configurations, including the four startup configurations, and completed seven. Its three later cohorts each contributed one full-budget survivor. Random with pruning completed two later cohorts and six models in total. Each full-training arm completed one four-candidate cohort after startup. Thus the pruning controller exercised staged allocation and admitted more candidates under the allocation policy.

MOTPE with pruning produced the largest observed final hypervolume at the stated reference scales. It also used the most active time. Together, hypervolume, candidate counts and time-progress curves describe the trade-off sets produced by the four executed searches; the experimental conditions are addressed in the Discussion.

Total active search time was 17.31 hours. There were 44 attempted trial entries across arms, including repeated shared configurations, and 29 eligible completed checkpoints. Only 15 distinct configurations appear among those completed checkpoints.

<!-- pagebreak -->

## 8 Accuracy and the observed Pareto set

### 8.1 Shared reference result

Every arm reached 92.12% best final validation accuracy because each included the same reference configuration. It uses stage depths 2-2-2-2, base width 32, AdamW learning rate 0.001, weight decay 0.0005, no dropout and basic augmentation. It has 2,797,610 parameters. Its clean-test accuracy is 91.47%; its average corrupted accuracy is 71.584%.

The highest final validation accuracy among later proposals was 91.02% for both random arms, 89.96% for MOTPE with pruning and 89.74% for MOTPE without pruning. The shared reference therefore remained the accuracy leader, while the completed archives supplied additional size and training-time trade-offs.

### 8.2 Main method trade-offs

![Figure 3 Pareto projections](../output/final-results/report-figures/pareto.png)

Figure 3. Two projections of completed MOTPE/pruning results. Pareto membership uses all three objectives together, so neither projection alone defines the full front. Model training minutes exclude validation and checkpoint overhead.

| Model ID prefix | Parameters | Final validation | Training minutes |
|---|---|---|---|
| 1c3e54237676 | 692,482 | 89.66% | 29.04 |
| e591e8832a2b | 1,575,202 | 90.44% | 27.12 |
| 13728d774901 | 2,797,610 | 92.12% | 24.78 |
| 260acdeb9f | 1,575,202 | 89.62% | 26.65 |
| 51817a4706 | 1,575,202 | 89.52% | 23.09 |

Table 2. The five observed Pareto models in the main method. Full configuration IDs and unrounded values are preserved in models.csv. The small model remains relevant because it uses fewer parameters. Another model trades accuracy for a shorter recorded training duration. Parameter count and laptop training speed need not order models in the same way.

Pareto membership is computed over eligible completed configurations using all three measured objectives. This observed set is the system output available for model selection.

<!-- pagebreak -->

## 9 Clean test and corruption robustness

### 9.1 Evaluation procedure

After each search was frozen, all eligible completed checkpoints were evaluated on 10,000 clean test examples and 75 corruption/severity conditions. Each condition contains 10,000 examples. The evaluator checks array shapes, data types and label correspondence before inference. It uses 15 standard corruptions: Gaussian, shot and impulse noise; defocus, glass, motion and zoom blur; snow, frost, fog, brightness, contrast, elastic transform, pixelation and JPEG compression [6].

Average corrupted accuracy is the arithmetic mean of the 75 condition accuracies. Mean corruption error is one minus that mean. Accuracy drop is clean accuracy minus mean corrupted accuracy, multiplied by 100 to report percentage points. This is raw mean error, not reference-normalized mCE. Neither clean-test nor corruption scores influence MOTPE, pruning or search-time Pareto selection.

| Arm | Mean clean accuracy | Mean corrupted accuracy |
|---|---|---|
| Random / full | 89.66% | 67.97% |
| MOTPE / pruning | 89.86% | 68.03% |
| Random / pruning | 90.35% | 68.70% |
| MOTPE / full* | 89.49% | 67.89% |

Table 3. Descriptive means over each arm's completed archive. Different archive sizes and configurations prevent treating these means as isolated method effects.

![Figure 4 Corruption severity](../output/final-results/report-figures/severity.png)

Figure 4. Archive mean accuracy falls as corruption severity increases. Each point averages the completed models and 15 corruption types at that severity. Each archive includes its shared startup configurations and later completed candidates.

The shared reference loses 19.886 percentage points from clean to average corrupted accuracy. All 29 reports are complete, giving 2,175 saved condition records and 22,040,000 image evaluations including clean data. Recorded evaluation time totals 86.88 minutes. The completed robustness analysis consists of per-condition metrics, accuracy drops and severity curves. No fitted robustness predictor or feature-importance result is reported.

### 9.2 Completed custom-data dashboard demonstration

A separate demonstration exercised the implemented dashboard on 400 generated RGB images labelled circle or square. The ZIP supplied 280 training, 60 validation and 60 test images. The run used CUDA, batch size 64, seed 42, a 12-epoch full budget, three warmup epochs and a 900-second active search allocation. The controller used four startup configurations and later 4/2/1 cohorts at cumulative budgets of 3/6/12 epochs. Calibration and final evaluation were outside the search allocation.

| Measure | Recorded result |
|---|---|
| Configurations tried | 140 |
| Completed at 12 epochs | 38 |
| Candidates stopped early | 102 |
| Later cohorts completed | 34 |
| Promotion decisions verified | 68 |
| Observed Pareto models | 3 |
| Active search time | 834.80 seconds (13 minutes 55 seconds) |
| Pareto validation accuracy | 100% for all three models |
| Pareto clean-test accuracy | 60/60 correct for each model |

Table 4. Recorded results of the custom-data dashboard demonstration, separate from the four CIFAR-10 searches.

The job reached complete status. All 38 completed checkpoint/evaluation identities were checked, all 68 promotion decisions were recomputed, and the final Pareto set was independently verified from the recorded objectives. Each of the three exported models matched its checkpoint and reproduced the saved clean-test result on CPU. Both supplied example images were classified correctly by all three exports.

This demonstration confirms the complete upload-to-model workflow using the same search engine with a dataset-specific classifier. Its simple synthetic classes and 60-image test set make it a functional demonstration; the CIFAR-10 comparison remains the main predictive and robustness experiment. The three Pareto models differ in size and recorded time, with timing differences of less than a quarter of a second on this small dataset. Custom corruption scores were not computed.

<!-- pagebreak -->

## 10 Implementation and verification

### 10.1 Software structure

| Module | Responsibility |
|---|---|
| config, model | Validate candidate/protocol settings; build ResNet and optimizer |
| data, folder_data | Prepare and validate manifests; construct isolated data loaders |
| trainer, storage | Train, measure, checkpoint, lock runs and handle manual sessions |
| sampler | Random proposals and completed-history MOTPE proposals |
| selection, controller | Non-dominated sorting, crowding, rung scheduling and promotion |
| ledger, search | SQLite archive, admission decisions, budget accounting and frozen results |
| evaluation, inference | Held-out evaluation, portable exports and image prediction |
| comparison, analysis | Hypervolume, comparisons and guarded post-search characterization |
| uploads, local_jobs, local_worker | Validate uploaded ZIPs; calibrate and run explicit local jobs |
| dashboard, cli | Local interface and explicit command-line operations |

The search ledger stores candidate configurations, observations, decisions and controller state. Candidate files retain epoch histories and checkpoints. Comparison functions check frozen search results against the ledger before generating final summaries. Existing output paths are protected against accidental evaluation overwrite.

### 10.2 Verification evidence

The final software check passed 62 automated tests. These cover configuration validation, complementary splits, model shapes and parameter caps, optimizer decay exclusions, promotion quotas and ties, mixed-budget rejection, hypervolume against an independent inclusion-exclusion calculation, atomic writes, locking and simulated search pause/resume. Additional checks cover custom class consistency, decoded-image overlap, unsafe uploads, train/test separation, clean evaluation, portable inference, explicit job start, calibration budget guards and simulated upload-to-export orchestration.

The automated tests use generated images, CPU inference and simulated training functions. They do not replace real training evidence. Manual GPU evidence includes the short pause/resume comparison, the reference/small/large full-budget pilots, the shortened integration search, four completed main searches and 29 held-out evaluations.

The local interface was checked in a browser with the real saved results. The main method displayed seven completed rows and five rows after Pareto filtering. Model export and local CPU prediction worked without JavaScript errors. The user then started and completed the real custom-data GPU demonstration through the dashboard. The candidate list displays the five most recently updated models, expands to all candidates on request, and preserves that view during automatic refresh. Exported predictions were also checked against the corresponding model calculation in a software test.

### 10.3 Audit corrections and preservation

Identical configuration/environment metadata does not guarantee identical trained weights. Post-search analysis now distinguishes checkpoint artifacts when hashes are available and retains separate legacy report paths while grouping by configuration. This prevents the observed repeated-configuration difference from being silently merged or treated as independent configuration diversity.

The original experiment source snapshot is retained. New dashboard, inference and custom-data functionality was added after the experiments. All 29 historical reports were rechecked for identity, completeness and metric consistency; their checkpoint and evaluation hashes were recorded. The original measurements and checkpoint identities are preserved alongside the current implementation.

<!-- pagebreak -->

## 11 Local use and supported custom data

### 11.1 Local dashboard and model use

Start the local interface with the following command from the project directory, then open http://127.0.0.1:8765. No further training is needed to demonstrate the completed experiments.

```text
.\.venv\Scripts\python.exe -m cs400.dashboard
```

The interface reads saved searches and displays their completed models, Pareto membership, validation and clean-test accuracy, corruption accuracy, parameter counts and measured training time. Selecting a model shows its configuration, learning curve and average corruption accuracy by severity. Export includes weights, configuration, class names and preprocessing metadata; optimizer state is omitted. A local image can be classified on CPU using that export.

The upload form validates a dataset before offering an explicit Start action. A separate local worker calibrates, searches, evaluates clean test accuracy after search is frozen, and exports Pareto models. Pause/Resume preserves epoch checkpoints. Merely opening the page or uploading data does not train. The server binds to loopback so the interface and execution remain on the user's computer.

### 11.2 Custom image-folder format

The dashboard accepts a ZIP of PNG/JPEG class folders, or explicit train, validation and test folders with the same class names. Class-only uploads need ten distinct images per class and receive a saved approximately 70/15/15 stratified split using seed 42. Explicit splits require an image per class in every split. Validation rejects unsafe ZIP paths and identical decoded images crossing splits or labels, and records image hashes. Limits include 512 MiB compressed, 2 GiB expanded, 100,000 images and 2 to 100 classes.

```text
data/my-dataset/train/class_a/image.png
data/my-dataset/validation/class_a/image.png
data/my-dataset/test/class_a/image.png
```

The example shows the directory pattern; each class needs distinct images in each split. Supported preprocessing converts to RGB, resizes directly to 32 by 32 using bilinear interpolation, and normalizes with fixed channel mean and standard deviation 0.5. Crop/flip and optional erasing remain training transformations. Preparation requires the user to confirm that these operations preserve the dataset's labels.

Search uses only training and validation loaders; clean evaluation loads the test split after completion. The classifier dimension follows the number of manifest classes. A changed image is rejected on subsequent dataset verification. Duplicate detection does not establish subject independence or detect all near-duplicates; appropriate source/subject splitting remains necessary.

### 11.3 Calibration and evaluation policy

Each new job begins with a three-epoch largest-model calibration and a 25% timing allowance. Search starts only if startup and one cohort are estimated to fit; otherwise the user revises the budget before search begins. Calibration and final evaluation are outside the active search allocation. The completed synthetic GPU demonstration verifies this orchestration. Clean evaluation uses the uploaded dataset's own test split. CIFAR-10-C remains specific to CIFAR-10; a custom corruption study requires a suitable domain benchmark.

<!-- pagebreak -->

## 12 Limitations and conclusions

### 12.1 Experimental limitations

The main comparison uses one seed per method, shared startup configurations and a common 5.25-hour cap. Actual search time ranges from 3.88 to 4.88 hours because admission stops when another estimated cohort will not fit. Accordingly, the report compares observed archives and time-progress curves under the executed conditions, rather than treating the results as repeated-seed or exactly equal-time estimates.

The MOTPE/full run includes a recorded battery-power interval during the reference model's epochs 36-60 and the next candidate's epochs 1-17, with reconnection during epoch 17. These durations were retained and affect its time objective, budget use and hypervolume. One repeated startup configuration also differed from epoch 51 onward in 122 of 146 model tensors, with a 0.01 percentage-point clean-test difference. Its cause remains unresolved. Separate checkpoint identities preserve the actual results of each run.

Intermediate promotion uses a three-epoch validation average and a linear time estimate. Final Pareto membership uses completed results only; fully training stopped configurations would be needed to measure promotion losses. The custom shapes demonstration verifies system operation on its supplied splits. It is reported separately from the CIFAR-10 benchmark and does not provide evidence of performance on other domains.

### 12.2 What the project establishes

The project delivers a working local system that combines joint ResNet configuration search, multi-objective promotion, checkpointed execution, held-out evaluation and model export. The four CIFAR-10 searches produced measured trade-off sets and complete corruption reports. The custom-data dashboard demonstration verified the full user workflow on a new two-class dataset, including training, evaluation, three Pareto exports and CPU inference.

MOTPE with pruning produced the largest observed hypervolume in the main comparison. The shared reference supplied the highest final validation accuracy, while the Pareto archive retained models with different size and recorded-time properties. The contribution is therefore both an implemented framework and a documented evaluation of its operation. Saved protocols, checkpoints, decisions and results provide concrete evidence for each stage.

### 12.3 Further work

The next experimental step is repeated-seed evaluation under controlled power conditions, together with investigation of the observed long-run checkpoint difference. Larger search archives would support more detailed study of proposal quality, promotion decisions and robustness associations. Additional datasets can extend validation of the upload workflow. Domain-specific corruption adapters, broader preprocessing policies and inference latency measurements are further development directions. These are proposed extensions to the completed system.

<!-- pagebreak -->

## 13 Reproduction and evidence files

| Artifact | Contents |
|---|---|
| configs/experiment-seed42.json | Frozen main training and search settings |
| data/cifar10-split.json | Saved split, preprocessing, class mapping and fingerprint |
| output/experiment-source-seed42.zip | Original source used for the main experiments |
| output/experiment-freeze-seed42.json | Experiment freeze record |
| runs/main-*/search.sqlite | Trial archive, decisions and controller state |
| runs/main-*/trials/* | Candidate metadata, histories, sessions and checkpoints |
| runs/main-*/evaluations/* | Clean and per-condition corruption reports |
| output/final-results | Joined CSV data, summary, comparisons and plots |
| output/final-verification | Checkpoint hashes, analysis and custom-dashboard-demo.json audit |
| runs/local-*/exports | Verified portable models from the custom dashboard demonstration |
| docs/LOCAL_DEMO.md | Local demonstration and custom-dataset commands |

Use the original source snapshot and recorded environment when reproducing the historical experiment. The current source fingerprint differs because product features were added afterward. Completed old checkpoints remain readable for evaluation/export, but the trainer rejects a silent resume with altered code. Portable model exports are intended for inference, while full checkpoints retain training state.

The evidence directory contains 29 joined model rows, 2,175 corruption-condition rows and 2,040 recorded epoch rows, including pruned candidates. CSV accuracies are fractions; report tables and axes display percentages. Preserving these raw records allows additional visualizations without retraining or changing the original measurements.

## References

[1] K. He, X. Zhang, S. Ren and J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016. https://arxiv.org/abs/1512.03385

[2] Optuna contributors. TPESampler documentation, version 4.5.0. Multi-objective TPE implementation and sampler settings. https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html

[3] K. Jamieson and A. Talwalkar. Non-stochastic Best Arm Identification and Hyperparameter Optimization. AISTATS, PMLR 51, 240-248, 2016. https://proceedings.mlr.press/v51/jamieson16.html

[4] K. Deb, A. Pratap, S. Agarwal and T. Meyarivan. A Fast and Elitist Multiobjective Genetic Algorithm NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197, 2002. https://doi.org/10.1109/4235.996017

[5] A. Krizhevsky, V. Nair and G. Hinton. CIFAR-10 and CIFAR-100 datasets. https://www.cs.toronto.edu/~kriz/cifar.html

[6] D. Hendrycks and T. Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR, 2019. https://arxiv.org/abs/1903.12261 . Benchmark implementation and datasets: https://github.com/hendrycks/robustness

[7] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. ICLR, 2019. https://arxiv.org/abs/1711.05101
