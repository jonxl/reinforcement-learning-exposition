"""The run driver: takes (algorithm, problem, number of seeds), runs the seeds, saves the results.

An algorithm module (``algos.<name>``) only knows how to train one batch of seeds:

    train(problem: envs.Problem, seeds: np.ndarray, cfg: Config) -> dict of per-seed arrays

with arrays shaped (n_seeds, n_checkpoints, ...) and a shared ``steps`` axis. The problem comes from
the registry in ``utils.envs`` by (name, gamma, params), so any algorithm runs on any environment and
the three values saved with a run are enough to rebuild its environment. Everything about
*running* lives here: drawing evaluation seeds, splitting them into chunks, running chunks in
parallel processes (processes, not threads: the work is many small numpy calls, which the GIL
would serialize), stitching the per-seed arrays back together, and saving them with enough
metadata (seeds, config, baseline) to reproduce any figure.

Because every seed draws its random numbers from its own generator, the result of a seed does not
depend on the chunking or on the number of workers.
"""

from __future__ import annotations

import json
import os
import time
import multiprocessing as mp
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from utils import envs, seeds
from utils.learning_curves import LearningCurves
from utils.devices import select_devices
from algos.families import VARIANTS, ALIASES, canonical_name, get_algorithm

RESULTS_DIR = Path("results")

ALGOS = dict(VARIANTS)
ALGO_CHOICES = tuple(VARIANTS) + tuple(ALIASES)
DEFAULT_BENCHMARK_ALGOS = ("ppo-nn", "monte_carlo")
PROBLEMS = tuple(envs.PROBLEMS)


def get_algo(name: str):
    return get_algorithm(name)


def problem_of(meta: dict) -> envs.Problem:
    """Rebuild the problem of a saved run from its metadata."""
    return envs.make_problem(meta["problem"], meta["gamma"], **meta.get("problem_params", {}))


def make_config(algo, overrides: dict | None):
    cfg = algo.Config()
    if overrides:
        known = {f.name for f in fields(cfg)}
        bad = set(overrides) - known
        if bad:
            raise KeyError(f"unknown config keys {sorted(bad)}; known: {sorted(known)}")
        cfg = replace(cfg, **overrides)
    return cfg


def _run_chunk(algo_name: str, problem_name: str, gamma: float, problem_params: dict, chunk: np.ndarray, cfg) -> dict:
    algo = get_algo(algo_name)
    algo.validate_problem(problem_name)
    return algo.train(envs.make_problem(problem_name, gamma, **problem_params), chunk, cfg)


def _gpu_batches(algo_name, problem_name, gamma, problem_params, chunks, cfg, devices):
    """One spawned process per GPU; each pulls another seed batch when idle."""
    parts = [None] * len(chunks)
    jobs = iter(enumerate(chunks))
    with ExitStack() as stack:
        pools = [(device, stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context('spawn')))) for device in devices]
        pending = {}

        def submit(device, pool):
            job = next(jobs, None)
            if job is None:
                return
            index, chunk = job
            future = pool.submit(_run_chunk, algo_name, problem_name, gamma,
                                 problem_params, chunk, replace(cfg, device=device))
            pending[future] = (index, device, pool)

        for device, pool in pools:
            submit(device, pool)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, device, pool = pending.pop(future)
                parts[index] = future.result()
                submit(device, pool)
    return parts


def run(
    algo_name: str,
    problem_name: str = "lqr",
    n_seeds: int = 32,
    master_seed: int = 0,
    gamma: float = 0.99,
    workers: int | None = None,
    overrides: dict | None = None,
    out_dir: Path | None = None,
    tag: str = "",
    problem_params: dict | None = None,
    device: str = "cpu",
    seed_batch_size: int | None = None,
) -> Path:
    """Run ``n_seeds`` evaluation seeds of one algorithm on one problem and save the results."""
    algo_name = canonical_name(algo_name)
    algo = get_algo(algo_name)
    algo.validate_problem(problem_name)
    if overrides and 'device' in overrides:
        raise ValueError('Select execution with --cpu/--gpu, not --set device=...')
    if n_seeds < 1 or (workers is not None and workers < 1):
        raise ValueError('seeds and workers must be positive')
    if seed_batch_size is not None and seed_batch_size < 1:
        raise ValueError('seed_batch_size must be positive')
    devices = select_devices(device, algo_name)
    cfg = make_config(algo, overrides)
    problem_params = dict(problem_params or {})
    problem = envs.make_problem(problem_name, gamma, **problem_params)
    eval_seeds = seeds.split_seeds(n_tune=0, n_eval=n_seeds, master_seed=master_seed).eval

    gpu = devices[0].startswith('cuda:')
    if gpu:
        batch_size = seed_batch_size or 16
        chunks = [eval_seeds[i:i+batch_size] for i in range(0, n_seeds, batch_size)]
        devices = devices[:len(chunks)]
        actual_workers = len(devices)
    else:
        actual_workers = workers or min(os.cpu_count() or 1, n_seeds)
        chunks = ([eval_seeds[i:i+seed_batch_size] for i in range(0, n_seeds, seed_batch_size)]
                  if seed_batch_size else [c for c in np.array_split(eval_seeds, actual_workers) if len(c)])
        actual_workers = min(actual_workers, len(chunks))
    print(f'{algo_name}: devices={devices}, workers={actual_workers}, seed batches={len(chunks)}', flush=True)
    t0 = time.perf_counter()
    if gpu:
        parts = _gpu_batches(algo_name, problem_name, gamma, problem_params, chunks, cfg, devices)
    elif actual_workers == 1:
        parts = [_run_chunk(algo_name, problem_name, gamma, problem_params, chunk, cfg) for chunk in chunks]
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = [pool.submit(_run_chunk, algo_name, problem_name, gamma, problem_params, c, cfg) for c in chunks]
            parts = [f.result() for f in futures]
    wall = time.perf_counter() - t0

    arrays = {"steps": parts[0]["steps"], "seeds": eval_seeds}
    for key, value in parts[0].items():
        if isinstance(value, np.ndarray) and key != "steps":
            arrays[key] = np.concatenate([p[key] for p in parts], axis=0)
    scalars = {k: v for k, v in parts[0].items() if not isinstance(v, np.ndarray)}
    if 'failure_records' in scalars:
        scalars['failure_records'] = [record for part in parts for record in part['failure_records']]

    meta = {
        "algo": algo_name,
        "algorithm_family": algo.family,
        "algorithm_variant": algo.variant,
        "problem": problem_name,
        "problem_params": problem_params,
        "gamma": gamma,
        "n_seeds": n_seeds,
        "master_seed": master_seed,
        "workers": actual_workers,
        "execution": {"requested": device, "devices": devices,
                      "seed_batch_size": max(map(len, chunks)), "batches": len(chunks)},
        "wall_seconds": wall,
        "J_star": problem.J_star,
        "K_star": problem.K_star.reshape(-1).tolist(),
        "problem_specification": getattr(problem, 'benchmark_specification', None),
        # K* is the optimum only where the optimal policy is a linear gain; the curvature of the gap
        # at K* (bias-variance split) exists only there. Elsewhere K* is the gain of the linearization.
        "curvature": (0.5 * problem.gap_hessian()).tolist() if hasattr(problem, "gap_hessian") else None,
        "created": datetime.now().isoformat(timespec="seconds"),
        **scalars,
    }
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = RESULTS_DIR / f"{stamp}_{algo_name}_{problem_name}{'_' + tag if tag else ''}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return out_dir


def load(run_dir: Path) -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    with np.load(run_dir / "arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads((run_dir / "meta.json").read_text())
    return arrays, meta


def to_curves(arrays: dict, meta: dict) -> LearningCurves:
    return LearningCurves(
        steps=arrays["steps"],
        gaps=_checked_gaps(arrays["gaps"]),
        gains=arrays.get("gains"),
        unstable=arrays.get("unstable"),
        name=meta["algo"],
        meta=meta,
    )


def _checked_gaps(gaps: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Clip round-off below zero; a clearly negative gap means J < J*, i.e. a bug or a truncated rollout."""
    if np.any(gaps < -tol):
        raise ValueError(f"gap below zero (min {gaps.min():.3g}): evaluation horizon too short or a bug")
    return np.clip(gaps, 0.0, 1.0)
