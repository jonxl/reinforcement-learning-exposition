"""Learning algorithms scored against the baselines (Monte Carlo control, PPO, ...).

Each algorithm will expose the same interface: train on an environment with one seed and return
its gains (or effective gains) and gaps at checkpoints counted in environment steps.
"""
