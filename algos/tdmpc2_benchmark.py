"""Compatibility imports for historical callers; use the explicit variant modules.

Family classes are exposed here, while compatible module attributes resolve lazily.
"""
from algos.families import TDMPC2LG


def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('algos.td_mpc2_lg'), name)
