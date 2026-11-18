"""High-level commands for the experiments.

    uv run main.py baseline --problem lqr --gamma 0.99     exact optimum the agents are scored against
    uv run main.py baseline --problem lqi --gamma 0.99
    uv run main.py baseline --problem nlqr --problem-set k=2
    uv run main.py budget --eps 0.05 --delta 0.05 --checkpoints 20    seeds needed for a confidence band

    uv run main.py crude --algos monte_carlo --problem lqr      small run + every metric and figure
    uv run main.py run --algo monte_carlo --problem lqr --seeds 1000 --set total_steps=2000000
    uv run main.py crude --algos monte_carlo ppo --problem nlqr --problem-set k=2
    uv run main.py report results/<run_a> results/<run_b> --out results/report_x

Problems (environments) are registered in ``utils.envs.PROBLEMS``; ``--problem-set KEY=VALUE`` sets
their parameters, e.g. the nonlinearity k of ``nlqr``. Algorithms are registered in ``rollouts.ALGOS``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import rollouts
from utils import baseline, concentration, envs


def cmd_baseline(args: argparse.Namespace) -> None:
    if args.problem == "lqr":
        prob = baseline.scalar_lqr(gamma=args.gamma)
        lo, hi = baseline.scalar_finite_cost_interval(args.gamma)
        print(f"Scalar LQR  (a = b = q = r = x0 = 1, gamma = {args.gamma})")
        print(f"  K*                     = {prob.K_star.item():.6f}")
        print(f"  J* = p                 = {prob.J_star:.6f}")
        print(f"  finite-cost interval   = ({lo:.6f}, {hi:.6f})")
        print(f"  curvature c_gamma      = {baseline.scalar_curvature(args.gamma):.6f}   (g ~ c e^2 near K*)")
    elif args.problem in ("nlqr", "nlqr_scalar"):
        prob = envs.make_problem(args.problem, args.gamma, **_parse_overrides(args.problem_set))
        gaps, unstable = envs.score_gains(prob, np.stack([prob.K_star, np.zeros_like(prob.K_star)]))
        print(f"Nonlinear benchmark {args.problem}  (k = {prob.k}, x0 = {prob.x0.tolist()}, gamma = {args.gamma})")
        print(f"  A (linearization)      = {np.round(prob.linear.A, 6).tolist()}")
        print(f"  B (linearization)      = {prob.linear.B.reshape(-1).tolist()},  open-loop spectral radius "
              f"{max(abs(np.linalg.eigvals(prob.linear.A))):.4f}")
        print(f"  P (exact, every k)     = {(np.round(prob.P, 6) + 0.0).tolist()}")
        print(f"  J* = x0' P x0          = {prob.J_star:.6f}")
        print(f"  K* of the linearization= {np.round(prob.K_star.reshape(-1), 6).tolist()}")
        print(f"  gap of u = -K* x       = {gaps[0]:.6f}   (unstable: {bool(unstable[0])})")
        print(f"  gap of u = 0           = {gaps[1]:.6f}   (unstable: {bool(unstable[1])})")
    else:
        prob = baseline.scalar_lqi(gamma=args.gamma)
        x_ss, u_ss = prob.equilibrium
        K_x, K_z = prob.K_star.reshape(-1)
        print(f"Scalar LQI  (a = b = c = q = q_z = r = 1, y_ref = 1, x0 = 0, d = 0, gamma = {args.gamma})")
        print(f"  K* = [K_x, K_z]        = [{K_x:.6f}, {K_z:.6f}]")
        print(f"  J*                     = {prob.J_star:.6f}")
        print(f"  equilibrium x_ss, u_ss = {x_ss.item():.6f}, {u_ss.item():.6f}")
        print(f"  closed loop stable     = {prob.is_stable(prob.K_star)}")


def cmd_budget(args: argparse.Namespace) -> None:
    n = args.checkpoints
    print(f"Seeds so that every one of {n} checkpoint(s) is within eps = {args.eps} "
          f"of the expected curve with probability >= {1 - args.delta}:")
    print(f"  Hoeffding                 M >= {concentration.hoeffding_seeds(args.eps, args.delta, n)}")
    print(f"  Chebyshev (Var <= 1/4)    M >= {concentration.chebyshev_seeds(args.eps, args.delta / n)}")
    if args.seeds:
        eps = concentration.hoeffding_eps(args.seeds, args.delta, n)
        print(f"  with M = {args.seeds}: Hoeffding half-width eps = {eps:.4f}")


def _parse_overrides(pairs: list[str]) -> dict:
    """--set key=value ... -> {key: value}, with ints, floats and booleans parsed."""
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
            continue
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def _report(run_dirs: list[Path], out: Path, delta: float) -> None:
    from utils import graphs  # matplotlib only when needed

    runs = []
    for d in run_dirs:
        arrays, meta = rollouts.load(d)
        runs.append((rollouts.to_curves(arrays, meta), arrays, meta))
    problems = {(meta["problem"], meta["gamma"], tuple(sorted(meta.get("problem_params", {}).items()))) for _, _, meta in runs}
    if len(problems) != 1:
        raise SystemExit(f"report needs runs on one problem with the same parameters, got {sorted(problems)}")
    floor, _ = envs.best_linear_gap(rollouts.problem_of(runs[0][2]))
    graphs.make_report(runs, out, delta, linear_floor=floor if floor > 0 else None)
    print((out / "summary.md").read_text())
    print(f"figures and summary written to {out}/")


def cmd_run(args: argparse.Namespace) -> None:
    out = rollouts.run(args.algo, args.problem, args.seeds, args.master_seed, args.gamma, args.workers,
                       _parse_overrides(args.set), tag=args.tag, problem_params=_parse_overrides(args.problem_set))
    print(f"saved {out}/")


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out) if args.out else rollouts.RESULTS_DIR / f"report_{datetime.now():%Y%m%d-%H%M%S}"
    _report([Path(d) for d in args.runs], out, args.delta)


def cmd_crude(args: argparse.Namespace) -> None:
    """A quick, small estimate: run each algorithm on a few seeds, then every metric and figure."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = rollouts.RESULTS_DIR / f"crude_{stamp}_{args.problem}"
    overrides = _parse_overrides(args.set)
    run_dirs = []
    for algo in args.algos:
        print(f"running {algo} on {args.problem}: {args.seeds} seeds ...", flush=True)
        d = rollouts.run(algo, args.problem, args.seeds, args.master_seed, args.gamma, args.workers,
                         overrides, out_dir=root / algo, problem_params=_parse_overrides(args.problem_set))
        run_dirs.append(d)
    _report(run_dirs, root / "report", args.delta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="print the exact optimum of a problem")
    p.add_argument("--problem", choices=list(rollouts.PROBLEMS), default="lqr")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--problem-set", nargs="*", default=[], metavar="KEY=VALUE", help="problem parameters")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("budget", help="number of seeds for a confidence band of given width and risk")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--checkpoints", type=int, default=1, help="union bound over this many checkpoints")
    p.add_argument("--seeds", type=int, default=0, help="also report the band width for this many seeds")
    p.set_defaults(func=cmd_budget)

    def add_run_args(p, multi: bool):
        if multi:
            p.add_argument("--algos", nargs="+", default=["monte_carlo"], choices=sorted(rollouts.ALGOS))
        else:
            p.add_argument("--algo", required=True, choices=sorted(rollouts.ALGOS))
        p.add_argument("--problem", choices=list(rollouts.PROBLEMS), default="lqr")
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--master-seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=None, help="processes (default: one per core, up to #seeds)")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="algorithm config overrides")
        p.add_argument("--problem-set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="problem parameters, e.g. k=3 for nlqr")

    p = sub.add_parser("run", help="run one algorithm on many seeds and save the learning curves")
    add_run_args(p, multi=False)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--tag", default="")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="figures and metrics from saved runs on one problem")
    p.add_argument("runs", nargs="+", help="run directories under results/")
    p.add_argument("--out", default=None)
    p.add_argument("--delta", type=float, default=0.05)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("crude", help="small run of each algorithm, then every metric and figure")
    add_run_args(p, multi=True)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--delta", type=float, default=0.05)
    p.set_defaults(func=cmd_crude)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    np.set_printoptions(precision=6)
    args.func(args)


if __name__ == "__main__":
    main()
