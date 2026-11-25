# Creating numerical-experiment reports

[README](../README.md) · [Running experiments](usage.md) · [Reading the graphs](reading-the-graphs.md)

## Generate and rebuild a benchmark report

`suite` and `matched-lq` automatically generate reports after training. Rebuild
from saved data without retraining:

```sh
uv run main.py report results/all_algorithms
# For the nonlinear suite:
uv run main.py report results/nonlinear_suite
```

These paths refer to the experiments in the [usage guide](usage.md). Each produces:

```text
<experiment>/benchmark_report/
  index.html                 # links to each fixed problem
  benchmark.tex              # complete LaTeX draft
  manifest.json              # report provenance
  <problem>/
    index.html               # figures and summary table
    report.json              # statistics and run metadata
    report.tex               # LaTeX section included by benchmark.tex
    single_seed.png/.pdf
    seeds_and_mean.png/.pdf
    mean_learning_curve.png/.pdf
    seed_averaging.png/.pdf
```

The `.png/.pdf` notation means each figure is saved in both formats. The four
views show a predetermined seed, every seed with its mean, algorithm means, and
averaging across seeds at a fixed checkpoint. Problems remain separate in both
HTML and LaTeX.

Use an explicit report directory to keep a separate generated version:

```sh
uv run main.py report results/all_algorithms --out results/report_draft
```

## Compile the LaTeX draft

Install `latexmk` and a TeX distribution providing `pdflatex`, including the
`geometry`, `graphicx`, `hyperref`, and `bookmark` packages. From the repository
root, in a Unix-like environment with Bash, `realpath`, and `flock`:

```sh
bash scripts/build-latex.sh results/all_algorithms/benchmark_report/benchmark.tex
```

The helper runs the required LaTeX passes, stores intermediate files in
`benchmark_report/.build/manual/`, and publishes the completed
`benchmark_report/benchmark.pdf`. It keeps manual builds separate from the
editor's continuous compiler. PDF compilation is a separate step from `report`.

For a portable draft, keep the LaTeX sources and their figures together and change
absolute includes to relative paths. The generator writes absolute paths to
per-problem `.tex` files and figure PDFs. After moving raw results to a different
location or machine, rebuild with `report` to refresh those paths before compiling.

## Turn the generated draft into an experiment write-up

The generated draft supplies setup, tables, and figures. Add the research
question, chosen budgets and variants, observations, interpretation, and
limitations when preparing it for colleagues. Keep training seeds distinct from
planner evaluation streams, and identify the exact problem and reference cost.

Regeneration overwrites generated report sources. Keep editorial changes in a
separate `.tex` file or copied report directory, for example under
`notes/numerical-experiments/<experiment>/`, before running `report` again.
A copied draft still needs valid figure/include paths. Preserve the raw arrays
and metadata alongside any shared report so the figures remain reproducible.

## Reports from individual runs

The quick-start `crude` command uses the fixed-run reporting path. It produces
PNG/PDF figures, `summary.json`, and `summary.md` under its `report/` directory;
it does not generate `benchmark.tex`.

To rebuild this style of report, replace the paths below with the run directories
printed by `run` or the algorithm subdirectories produced by `crude`:

```sh
uv run main.py report "path/to/monte_carlo_run" "path/to/ppo_run" \
  --out results/fixed_problem_report
```

These are shell placeholders, not literal directory names. All inputs must use
the same problem, parameters, and discount. Use `suite` or `matched-lq` for the
automatic LaTeX workflow, or include these figure PDFs in a handwritten `.tex`
report.

## Historical bounded-comparison report

The historical comparison has a separate, experiment-specific pipeline. Its
`ppo` method is the original bounded linear PPO implementation, not the current
`ppo` alias for neural PPO. See the [reproduction instructions](../experiments/lq-comparison-README.md)
and [locked protocol](../experiments/lq-comparison-protocol.md) for training
commands and pinned dependencies.

Once the two full runs and both ablations exist at their canonical paths:

```sh
uv run --extra lq python -m experiments.analyze_lq_comparison
uv run --extra lq python -m experiments.report_lq_comparison
bash scripts/build-latex.sh output/latex/ppo-tdmpc2-lq-comparison/main.tex
```

These commands use the current project environment. For the original measured
software versions, run the Python modules in the separately pinned reproduction
environment described in the linked instructions.

The analysis expects `results/lq_comparison/{ppo,td}/`, 40 seeds per method,
one million interactions, and checkpoints every 50,000 steps. It also reads
`results/lq_fast_ablation/td/` and `results/lq_structure_ablation/td/` plus saved
TD weights. It writes figures, `summary.json`, and `checkpoint-summary.csv` under
`output/latex/ppo-tdmpc2-lq-comparison/`. The report module reads that summary and
writes `report.md` and `main.tex`; its narrative is specific to this historical
experiment. Use the shared benchmark report for new configurations.

The build helper writes `main.pdf` beside `main.tex`. It does not update the
separately archived `output/pdf/ppo-tdmpc2-lq-comparison.pdf` automatically.

To import the historical raw comparison into the common report format:

```sh
uv run main.py report results/lq_comparison
```

This writes converted data and reports under `results/lq_comparison/unified/`.

## Compile an existing numerical-experiment write-up

The same helper accepts any existing `.tex` document, for example:

```sh
bash scripts/build-latex.sh \
  notes/numerical-experiments/first-ppo-lqr-baseline/first-ppo-lqr-baseline-test.tex
```

The completed PDF appears next to its source. This compiles the existing write-up
and its figures; it does not rerun training or update the scientific narrative.
