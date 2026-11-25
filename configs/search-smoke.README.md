# Automated search integration check

`search-smoke.json` is a short software integration test on CIFAR-10. It is
not the final research protocol and its results must not be pooled with the
60-epoch pilots or reported as evidence of search quality.

The test uses two startup candidates trained to nine epochs, followed by
MOTPE cohorts of four candidates at three epochs, two at six epochs, and
one at nine epochs. These are cumulative budgets: promoted candidates resume
their checkpoints. The existing five-epoch warmup is retained, so the first
test rung occurs during warmup; this schedule checks execution only, not the
quality of early selection. Two startup observations exercise the sampler's
model-based path but are insufficient to judge optimization effectiveness.

The `pilot_confirmed` flag authorizes only this separate integration-test
configuration after the manual GPU pilots; it does not mark the research
protocol as calibrated. `configs/pilot.json` remains unchanged.

The overall active-time allowance is 1,800 seconds, excluding pauses between
sessions. The controller can stop earlier if the estimated remaining budget
cannot accommodate another cohort or rung. Use 45 seconds per model-epoch
as a provisional admission estimate, with overhead allowance based on the
observed pilot speeds. It is not a guaranteed upper bound. Actual active time
controls the cutoff; a running epoch can overrun it.

Startup plus one halving cohort requires 39 model-epochs; at the admission
estimate this is 1,755 seconds. More cohorts can run only if the measured
remaining budget permits them. Failed candidates can reduce survivor counts.

Run manually from the project root:

```powershell
.\.venv\Scripts\python.exe -m cs400 search --protocol configs/search-smoke.json --run-dir runs/search-smoke --session-minutes 15 --epoch-cost-seconds 45 --start-training
```

For a normal session pause, repeat with `--resume`. Wait for the command to
return before shutting down. Unexpected termination of an active search
requires accounting review; do not bypass the controller's reconciliation
guard. No training is started by creating this configuration or running `plan`.

After execution, check the stored cohort promotions, checkpoint continuation,
completed-only archive, Pareto results, and active-time accounting. The test
does not evaluate clean test data or CIFAR-10-C. Final 15/30/60-epoch research
settings, the research search budget, pruning quality, and corruption evaluation
still require separate validation.
