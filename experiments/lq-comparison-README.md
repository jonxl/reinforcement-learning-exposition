# Reproducing the bounded comparison

The experiment uses the scalar plant and PPO training settings from the initial report, rerunning both algorithms with applied actions in [-2,2]. Read [the locked protocol](lq-comparison-protocol.md) for the exact estimand, tail proof, failure policy and differences from the historical unbounded PPO report. TD-MPC2-LQ is the fast structural adaptation, not standard neural TD-MPC2.

From the repository root, use a Python 3.12 environment with `pip install -r experiments/lq-requirements.txt`. The measured environment was `.venv-lq`; the separate original neural environment is unchanged. The root package also exposes an optional `lq` extra for current dependencies, but changing versions may change numerical trajectories; the measured version pins above are the reproduction target.

```powershell
.venv-lq\Scripts\python.exe -m pytest tests -q
.venv-lq\Scripts\python.exe -m experiments.lq_comparison --method ppo --seeds 40 --steps 1000000 --out results/lq_comparison
.venv-lq\Scripts\python.exe -m experiments.lq_comparison --method td --seeds 40 --steps 1000000 --workers 4 --out results/lq_comparison
.venv-lq\Scripts\python.exe -m experiments.lq_comparison --method td --seeds 4 --steps 10000 --checkpoints 2 --out results/lq_fast_ablation
.venv-lq\Scripts\python.exe -m experiments.lq_comparison --method td --seeds 4 --steps 10000 --checkpoints 2 --structure-only --out results/lq_structure_ablation
.venv-lq\Scripts\python.exe -m experiments.analyze_lq_comparison
.venv-lq\Scripts\python.exe -m experiments.report_lq_comparison
```

Existing output manifests are protected from overwriting: archive outputs or choose another `--out` path for a rerun. The analysis defaults to the canonical output paths. The matched comparison uses 80 million total training interactions, 40 million for each algorithm. Ablations, throughput runs and evaluations are additional work and are not counted as primary training interactions.

TD seed groups are independent OS processes, not interacting agents. A separate four-seed 4,000-step serial/parallel check produced bit-identical parameters, target critics, optimizer moments and warm starts at the final checkpoint. The evaluation horizon may differ between batch sizes because stopping requires every member's tail to be below tolerance; this changes only the negligible certified truncation interval.

Raw manifests, arrays, per-checkpoint weights, source snapshots, failures and final TD RNG states live under `results/`. These are intentionally ignored by Git. Per-shard files preserve the training order and individual progress. Checkpoints do **not** contain full replay, so they support rescoring rather than exact training resumption. The final model files do not by themselves reproduce training without rerunning the seed/configuration/source combination. Preserve raw outputs when sharing results.

The main analysis saves figures, a machine-readable summary and a checkpoint table under `output/latex/ppo-tdmpc2-lq-comparison/`. Its paired bootstrap uses 50,000 resamples and seed20261123, conditional on the three fixed planner evaluation streams. The conservative Hoeffding interval is reported alongside it; bootstrap precision must not be presented as a distribution-free guarantee.

Build the shareable report with two passes of `pdflatex -interaction=nonstopmode -halt-on-error main.tex` in that output directory. The Markdown report remains the native explanation; the LaTeX source and compiled PDF provide the presentation format used for the preceding experimental report.

## Integration with the common benchmark pipeline

`algos/ppo.py` remains neural PPO. The published comparison's original linear PPO
is preserved in `algos/ppo_lq_legacy.py`; its reproduction command still uses
`--method ppo`. This historical label must not be interpreted as neural PPO.

For a new matched experiment using the shared runner, seeds and reports:

```bash
uv run --extra lq main.py matched-lq --seeds 8 --steps 100000 --checkpoints 10 \
  --workers 4 --algos ppo monte_carlo tdmpc2_lq ppo_linear \
  --out results/matched_lq_new
```

Neural PPO, Monte Carlo and TD-MPC2-LQ are the defaults; linear PPO is optional.
All methods use the fixed scalar plant, start, discount and applied action bound
from the locked protocol. PPO still computes likelihoods for latent Gaussian
samples; the environment clips the physical actions. Evaluation uses actual
controllers and upper infinite-horizon cost bounds, not fitted-gain costs.
The new neural-PPO configuration uses 250-step rollouts, eight environments and
500-sample minibatches; other settings are saved in each run's metadata. Methods
have equal physical interaction budgets, not equal optimizer work or reset schedules.
Training budgets must be multiples of 2000. TD warmup is 2500 interactions, so
shorter runs test plumbing rather than learned planner performance.

This command creates the same `problems.json`, per-algorithm `arrays.npz` and
`meta.json`, and `benchmark_report/` structure as generated NLQR experiments.
Reports include individual runs, all seeds and means, mean comparisons, and
fixed-checkpoint averaging across seeds. TD evaluation is conditional on the
three fixed planner streams; those streams are not extra training seeds.

Convert the published raw results into this report format without retraining:

```bash
uv run main.py report results/lq_comparison
```

Converted arrays and reports are written under `results/lq_comparison/unified/`;
the original results and published report are retained. The algorithm ID
`ppo_linear_legacy` identifies the published PPO results. Rebuild any new matched
experiment with `uv run main.py report results/matched_lq_new`.

TD-MPC2-LQ currently supports only `lqr_bounded`. It is a 16-parameter scalar
structural adaptation, not neural TD-MPC2. It is deliberately rejected by the
generated two-state NLQR suite. Extending the model, planner and validation to
those problems is separate work; the generated NLQR benchmark remains unchanged.
