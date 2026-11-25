# Running experiments

[README](../README.md) · [Reading the graphs](reading-the-graphs.md) · [Creating reports](reports.md)

## Install

Run commands from the repository root with Python 3.12+ and `uv` installed:

```sh
uv sync --extra lq
```

The `lq` extra adds Numba for structured TD-MPC2. `uv sync` alone is sufficient
for the other variants. The project uses NumPy, SciPy, Matplotlib, and PyTorch;
the default PyTorch source is CPU-only.

## Start with one run

```sh
uv run main.py baseline --problem lqr
uv run main.py crude --algos monte_carlo --problem lqr \
  --seeds 1 --workers 1 \
  --set total_steps=10000 n_checkpoints=10
```

`baseline` prints the exact optimal cost and controller. `crude` trains, saves
results, and generates figures and a summary. Its output is
`results/crude_<timestamp>_lqr/`, with `monte_carlo/arrays.npz`,
`monte_carlo/meta.json`, and a `report/` directory. Start with
`report/monte_carlo_runs.png`. One seed illustrates a trajectory; it cannot measure
variation across training runs.

To compare Monte Carlo and linear-Gaussian PPO across seeds:

```sh
uv run main.py crude --algos monte_carlo ppo-lg --problem lqr \
  --seeds 8 --workers 4 \
  --set total_steps=100000 n_checkpoints=10
```

Use `run --algo ...` instead of `crude --algos ...` to save one algorithm without
generating a report. Rebuild figures later with `report`, passing the printed run
directory. Runs combined in one report must have the same problem, parameters,
and discount.

## All algorithms on one matched problem

```sh
uv run --extra lq main.py matched-lq \
  --algos monte_carlo ppo-nn ppo-lg td-mpc2-nn td-mpc2-lg \
  --seeds 40 --steps 1000000 --checkpoints 20 \
  --workers 4 --master-seed 0 \
  --out results/all_algorithms
```

This uses bounded scalar LQR: initial state 1, discount 0.99, dynamics
`x_next = x + u`, stage cost `x² + u²`, and applied actions clipped to `[-2, 2]`.
The exact LQR optimum is attainable from this initial state under those bounds.
Methods receive the same physical interaction budget per seed, although their
optimizer work, resets, and planning costs differ.

The full command uses 200 million training interactions plus evaluation. Try
`--seeds 3 --steps 10000 --checkpoints 5 --out results/all_algorithms_trial`
first for a smaller experiment. Matched budgets must be positive multiples
of 2,000. Structured TD-MPC2 starts with 2,500 warmup interactions.

The command automatically creates an HTML and LaTeX report under
`results/all_algorithms/benchmark_report/`.

## Generated nonlinear suite

```sh
uv run main.py suite \
  --algos monte_carlo ppo-nn ppo-lg td-mpc2-nn \
  --count 10 --seeds 40 --steps 102400 --checkpoints 20 \
  --workers 4 --generator-seed 0 --master-seed 0 \
  --out results/nonlinear_suite
```

This runs one scalar LQR reference and ten generated nonlinear problems with known
optima. Each problem gets its own seed averages and report; results are not pooled
across problems. This allocates 180,224,000 training interactions plus evaluation.
Suite budgets must be positive multiples of 512. For a small trial, use
`--count 1 --seeds 2 --steps 4096 --checkpoints 4` and a fresh output path.

`--generator-seed` fixes the problem collection; `--master-seed` fixes the training
seed draws. `--profile moderate` is the default generator profile; `aggressive`
selects the alternative profile. Saved `problems.json` records the actual problem
parameters and exact references.

## Algorithm choices

| CLI name | Implementation | Matched bounded LQR | Generated suite |
| --- | --- | --- | --- |
| `monte_carlo` | Tabular first-visit Monte Carlo control | Yes | Yes |
| `ppo-nn` | Neural Gaussian actor and neural value critic | Yes | Yes |
| `ppo-lg` | Linear-Gaussian actor and quadratic-feature critic | Yes | Yes |
| `td-mpc2-nn` | Neural encoder, dynamics, cost model, actor prior, and distributional critics | Yes | Yes |
| `td-mpc2-lg` | Scalar structured models and a linear-Gaussian actor prior | Yes | No |

The TD-MPC2 variants are project adaptations, not exact upstream reproductions.
The neural variant uses the problem's discount and imposes no artificial action
bounds on unbounded nonlinear problems. The structured variant supports only
`lqr_bounded`. PPO variants differ in critic and implementation as well as actor.

Legacy aliases remain accepted: `ppo` means `ppo-nn`; `ppo_linear` and `ppo-lq`
mean `ppo-lg`; `tdmpc2_lq` and `td-mpc2-lq` mean `td-mpc2-lg`. Historical reports
may use older labels; consult their recorded setup before comparing results.

## Configuration and seeds

| Option | Meaning |
| --- | --- |
| `--seeds` | Independent training runs per algorithm and problem |
| `--master-seed` | Seed used to draw training seed identifiers |
| `--steps` | Training interactions per seed in `suite` and `matched-lq` |
| `--checkpoints` | Evaluation schedule setting in `suite` and `matched-lq` |
| `--workers` | CPU worker processes |
| `--seed-batch-size` | Seeds per worker job, independent of total seed count |
| `--out` | New experiment directory for `suite` and `matched-lq` |

For `run` and `crude`, use `--set total_steps=... n_checkpoints=...` for algorithm
settings and `--problem-set key=value` for problem parameters. Overrides must be
fields supported by the chosen algorithm; shared overrides in `crude` must work
for every selected algorithm. These commands also accept `--gamma` and `--problem`.
Use `uv run main.py <command> --help` for command-specific options.

Seed batching controls execution rather than sample count. Each seed has its own
random stream. Keep configuration, seed identifiers, problem parameters, and
software versions with results you want to reproduce.

## CPU and CUDA

CPU is the default. Training commands accept mutually exclusive `--cpu` and
`--gpu`. GPU execution requires a CUDA-enabled PyTorch installation and a working
NVIDIA driver; the repository's default CPU wheel does not provide CUDA.
`--gpu` discovers visible CUDA devices and falls back to CPU with a message if
CUDA is unavailable. Structured variants and Monte Carlo retain CPU backends.

```sh
uv run main.py suite --algos ppo-nn td-mpc2-nn \
  --count 2 --seeds 8 --steps 10240 \
  --gpu --seed-batch-size 4 --out results/gpu_trial
```

CUDA uses one spawned worker per device with work; `--workers` controls CPU
execution. The default CUDA seed batch size is 16. Neural PPO batches independent
parameters across seeds; neural TD-MPC2 runs seeds sequentially within each job.
Environment simulation and some planner/replay operations still use the CPU.
Float64 precision is retained, so measure speed on your hardware. There is no
automatic batch-size tuning or out-of-memory retry. Metadata records requested
and actual execution devices.

## Saved results and reruns

Suite and matched runs refuse an existing output directory. Choose a new path
for another experiment. Each contains:

- `problems.json`: problem collection, algorithms, seeds, and experiment settings.
- `<problem>/<algorithm>/arrays.npz`: training checkpoints and per-seed metrics.
- `<problem>/<algorithm>/meta.json`: configuration, seeds, reference, and execution metadata.
- `benchmark_report/`: HTML, figures, statistics, and LaTeX source.

The common runner saves after an entire algorithm run completes. It does not
provide incremental persistence or training resume. Preserve raw `results/`
directories when sharing an experiment: they are ignored by Git, and figures
alone are insufficient to regenerate the analysis.

See [creating reports](reports.md) to rebuild outputs without retraining.
