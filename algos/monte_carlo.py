"""On-policy first-visit Monte Carlo control with an epsilon-greedy policy (Sutton & Barto, Sec. 5.4).

The crude baseline: a lookup table Q[state bin, action], no bootstrapping, no function approximation.

* State: each coordinate is binned on its own grid; the bins are finer near zero and an odd count
  puts one bin centred on zero (otherwise the bin next to zero can map to u = 0 and freeze the state,
  a "dead zone" that costs x^2 / (1 - gamma) forever).
* Action: a fixed grid containing 0 and log-spaced magnitudes on each side.
* Episodes: ``horizon`` steps from x0; the return of every first visit to (s, a) is averaged into
  Q(s, a) with the sample-average step 1/n.
* Behaviour: epsilon-greedy in Q, ties broken at random. Scoring: the greedy policy, by a long
  deterministic rollout (``utils.envs``), so the learning curve contains no exploration noise.

Seeds: every random number of seed omega comes from ``rng_for(omega)``, so a seed's run does not
depend on which other seeds share the batch. The batch over seeds is for speed only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from utils import envs
from utils.seeds import rng_for

NAME = "monte_carlo"


@dataclass
class Config:
    n_bins: int = 41  # per state coordinate; odd, so one bin is centred on zero
    state_limit: float = 2.0  # grid covers [-limit, limit]; states beyond fall in the edge bins
    bin_concentration: float = 3.0  # sinh spacing: larger -> finer near zero
    n_action_magnitudes: int = 30  # actions: 0 and +/- geomspace(action_min, action_max, this)
    action_min: float = 1e-3
    action_max: float = 2.0
    epsilon: float = 0.1
    horizon: int = 500  # training episode length; gamma^500 ~ 0.007 at gamma = 0.99
    total_steps: int = 1_000_000  # environment steps per seed
    n_checkpoints: int = 20
    eval_horizon: int = envs.EVAL_HORIZON


def state_edges(cfg: Config) -> np.ndarray:
    if cfg.n_bins % 2 == 0:
        raise ValueError("n_bins must be odd so that one bin is centred on zero")
    v = np.linspace(-1.0, 1.0, cfg.n_bins + 1)
    s = cfg.bin_concentration
    return cfg.state_limit * np.sinh(s * v) / np.sinh(s)


def action_grid(cfg: Config) -> np.ndarray:
    mags = np.geomspace(cfg.action_min, cfg.action_max, cfg.n_action_magnitudes)
    return np.concatenate([-mags[::-1], [0.0], mags])


class Table:
    """Maps continuous observations to flat state indices and holds Q and the visit counts."""

    def __init__(self, cfg: Config, n_state: int, n_seeds: int):
        self.edges = state_edges(cfg)
        self.actions = action_grid(cfg)
        self.n_state = n_state
        self.n_bins = cfg.n_bins
        self.n_cells = cfg.n_bins**n_state
        self.n_actions = len(self.actions)
        self.Q = np.zeros((n_seeds, self.n_cells, self.n_actions))
        self.N = np.zeros((n_seeds, self.n_cells, self.n_actions), dtype=np.int64)

    def index(self, obs: np.ndarray) -> np.ndarray:
        """obs (E, n_state) -> flat bin index (E,)."""
        b = np.clip(np.searchsorted(self.edges, obs, side="right") - 1, 0, self.n_bins - 1)
        flat = np.zeros(obs.shape[0], dtype=np.int64)
        for d in range(self.n_state):
            flat = flat * self.n_bins + b[:, d]
        return flat

    def greedy(self, s: np.ndarray, tie_noise: np.ndarray | None = None) -> np.ndarray:
        q = self.Q[np.arange(len(s)), s]
        if tie_noise is not None:
            q = q + tie_noise
        return q.argmax(axis=1)


def train(problem: envs.Problem, seeds: np.ndarray, cfg: Config) -> dict:
    """Train one table per seed; return checkpoints and per-seed metrics.

    ``problem`` is any problem of ``utils.envs``. The algorithm uses it only through the
    environment; J* and the gap are computed for scoring.
    """
    if problem.n_control != 1:
        raise ValueError("the action grid is scalar")
    M = len(seeds)
    rngs = [rng_for(s) for s in seeds]
    env = envs.Env(problem, M)
    table = Table(cfg, problem.n_state, M)
    nA, T, gamma = table.n_actions, cfg.horizon, problem.gamma
    # random tie-breaking among equal Q-values: a fixed tiny preference per seed, from its own stream
    tie_pref = np.stack([g.random(nA) * 1e-9 for g in rngs])

    n_episodes = max(1, cfg.total_steps // T)
    checkpoint_eps = np.unique(np.linspace(0, n_episodes, cfg.n_checkpoints + 1).astype(int))
    out = {k: [] for k in ("steps", "gaps", "gains", "unstable", "visited_frac", "final_abs_state", "diverged")}

    def score(episode: int) -> None:
        def greedy_policy(obs):
            return table.actions[table.greedy(table.index(obs), tie_pref)][:, None]

        s = envs.score_policy(problem, greedy_policy, M, cfg.eval_horizon)
        out["steps"].append(episode * T)
        for key in ("gaps", "gains", "unstable", "final_abs_state", "diverged"):
            out[key].append(s[key])
        out["visited_frac"].append((table.N.sum(axis=2) > 0).mean(axis=1))

    score(0)
    s_buf = np.empty((T, M), dtype=np.int64)
    a_buf = np.empty((T, M), dtype=np.int64)
    r_buf = np.empty((T, M))
    for episode in range(1, n_episodes + 1):
        # each seed's random numbers for this episode, from its own stream
        explore = np.empty((T, M), dtype=bool)
        rand_a = np.empty((T, M), dtype=np.int64)
        for m, g in enumerate(rngs):
            explore[:, m] = g.random(T) < cfg.epsilon
            rand_a[:, m] = g.integers(0, nA, T)

        obs = env.reset()
        for t in range(T):
            s = table.index(obs)
            a = np.where(explore[t], rand_a[t], table.greedy(s, tie_pref))
            obs, r = env.step(table.actions[a][:, None])
            s_buf[t], a_buf[t], r_buf[t] = s, a, r

        # returns G_t = r_t + gamma G_{t+1}, then first-visit sample averages
        G = np.empty((T, M))
        acc = np.zeros(M)
        for t in range(T - 1, -1, -1):
            acc = r_buf[t] + gamma * acc
            G[t] = acc
        key = s_buf * nA + a_buf  # (T, M) flat (state, action) per step
        first = np.zeros((T, M), dtype=bool)
        for m in range(M):
            _, idx = np.unique(key[:, m], return_index=True)
            first[idx, m] = True
        t_idx, m_idx = np.nonzero(first)
        s_up, a_up, g_up = s_buf[t_idx, m_idx], a_buf[t_idx, m_idx], G[t_idx, m_idx]
        table.N[m_idx, s_up, a_up] += 1
        n = table.N[m_idx, s_up, a_up]
        table.Q[m_idx, s_up, a_up] += (g_up - table.Q[m_idx, s_up, a_up]) / n

        if episode in checkpoint_eps:
            score(episode)

    result = {k: np.stack(v, axis=1) if k != "steps" else np.asarray(v) for k, v in out.items()}
    result["config"] = asdict(cfg)
    result["n_cells"] = table.n_cells
    result["n_actions"] = nA
    return result
