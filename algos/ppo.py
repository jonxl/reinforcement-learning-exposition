"""Compatibility imports for historical callers; use the explicit variant modules.

Family classes are exposed here, while compatible module attributes resolve lazily.
"""
from algos.families import PPO, PPONN, PPOLG


def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('algos.ppo_nn'), name)
