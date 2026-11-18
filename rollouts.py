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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np

from utils import envs, seeds
from utils.learning_curves import LearningCurves

RESULTS_DIR = Path("results")

ALGOS = {"monte_carlo": "algos.monte_carlo", "ppo": "algos.ppo", "ppo_linear": "algos.ppo_linear"}
PROBLEMS = tuple(envs.PROBLEMS)


def get_algo(name: str):
    if name not in ALGOS:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(ALGOS)}")
    return import_module(ALGOS[name])


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
    return algo.train(envs.make_problem(problem_name, gamma, **problem_params), chunk, cfg)


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
) -> Path:
    """Run ``n_seeds`` evaluation seeds of one algorithm on one problem and save the results."""
    algo = get_algo(algo_name)
    cfg = make_config(algo, overrides)
    problem_params = dict(problem_params or {})
    problem = envs.make_problem(problem_name, gamma, **problem_params)
    eval_seeds = seeds.split_seeds(n_tune=0, n_eval=n_seeds, master_seed=master_seed).eval

    workers = workers or min(os.cpu_count() or 1, n_seeds)
    chunks = [c for c in np.array_split(eval_seeds, workers) if len(c)]
    t0 = time.perf_counter()
    if len(chunks) == 1:
        parts = [_run_chunk(algo_name, problem_name, gamma, problem_params, chunks[0], cfg)]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(_run_chunk, algo_name, problem_name, gamma, problem_params, c, cfg) for c in chunks]
            parts = [f.result() for f in futures]
    wall = time.perf_counter() - t0

    arrays = {"steps": parts[0]["steps"], "seeds": eval_seeds}
    for key, value in parts[0].items():
        if isinstance(value, np.ndarray) and key != "steps":
            arrays[key] = np.concatenate([p[key] for p in parts], axis=0)
    scalars = {k: v for k, v in parts[0].items() if not isinstance(v, np.ndarray)}

    meta = {
        "algo": algo_name,
        "problem": problem_name,
        "problem_params": problem_params,
        "gamma": gamma,
        "n_seeds": n_seeds,
        "master_seed": master_seed,
        "workers": len(chunks),
        "wall_seconds": wall,
        "J_star": problem.J_star,
        "K_star": problem.K_star.reshape(-1).tolist(),
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
