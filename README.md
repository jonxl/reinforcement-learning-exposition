# reinforcement-learning-exposition
A new repo on a wild card idea for RL agents. 

## Algorithm variants

Use `ppo-nn` for a neural Gaussian actor and neural value critic, or `ppo-lg`
for a linear-Gaussian actor and quadratic-feature critic. `td-mpc2-nn` uses
neural encoder, dynamics, cost prediction, actor prior, and distributional
critics. `td-mpc2-lg` is the bounded scalar adaptation with structured models.
The family classes share the experiment interface; their architectures and
optimizer settings remain explicit in saved configurations.

Generate fixed NLQR problems and train both neural variants on each problem:

```sh
uv run main.py suite --algos ppo-nn td-mpc2-nn monte_carlo --count 10 --seeds 8 --steps 10240 --out results/neural_nlqr
uv run main.py report results/neural_nlqr
```

`report` only rebuilds saved results. The neural TD implementation is a CPU
benchmark variant, with problem-specific discount and no artificial action
bounds on unbounded NLQR problems; it is not an exact upstream reproduction.
Old algorithm spellings remain accepted for existing scripts and saved runs.

## CPU and CUDA execution

Training commands (`run`, `crude`, `suite`, `matched-lq`) accept mutually exclusive
`--cpu` and `--gpu` flags. CPU is the default. `--gpu` discovers visible CUDA
GPUs and falls back to CPU with a message when CUDA is unavailable. Structured
LG variants and Monte Carlo continue to use their CPU backends.

```sh
uv run main.py run --algo ppo-nn --problem nlqr_energy --seeds 1000 --gpu --seed-batch-size 16
uv run main.py suite --algos ppo-nn td-mpc2-nn ppo-lg monte_carlo --count 10 --seeds 100 --gpu --seed-batch-size 16 --out results/cuda_suite
uv run main.py run --algo ppo-lg --seeds 100 --cpu --workers 4 --seed-batch-size 8
```

Each detected GPU gets one spawned worker. Workers take seed batches from a
shared coordinator as they finish; total seeds and batch size are independent.
PPO batches independent actor/critic parameters over seeds. Neural TD-MPC2 runs
seeds sequentially within a job, with its neural networks and candidate
computations on the selected GPU. CPU environment simulation, NumPy random
streams, replay storage and planner sampling still involve host/device transfers.
Float64 precision is unchanged. GPU speedups are not assumed; benchmark on the
target hardware. CUDA requires a CUDA-enabled PyTorch installation and a working
NVIDIA driver. Saved metadata records requested and actual execution devices.

`--workers` controls CPU execution; CUDA uses one worker per device with work.
Batch size is explicit (CUDA default 16), without automatic memory tuning or
out-of-memory retries. The runner currently saves after the entire algorithm run
completes; incremental persistence and resume are still needed for long jobs.
