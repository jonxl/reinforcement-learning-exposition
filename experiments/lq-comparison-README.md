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
