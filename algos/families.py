"""Algorithm families and explicitly named representation variants.

Classes share the runner contract; numerical backends retain their tested updates.
Inheritance here does not assert that different variants have identical critics or
optimizers. Historical modules remain importable for experiment reproduction.
"""
from abc import ABC, abstractmethod
from importlib import import_module


class Algorithm(ABC):
    family = ''
    variant = ''
    name = ''
    backend = ''
    supported_problems = None
    available = True

    def require_available(self):
        if not self.available:
            raise NotImplementedError(f'{self.name} is not implemented; it is not an alias for a structured variant')

    @property
    def Config(self):
        self.require_available()
        return import_module(self.backend).Config

    def validate_problem(self, name):
        self.require_available()
        if self.supported_problems is not None and name not in self.supported_problems:
            raise ValueError(f'{self.name} supports only {self.supported_problems}; use matched-lq')

    @abstractmethod
    def train(self, problem, seeds, cfg):
        """Return the shared checkpoint arrays and configuration metadata."""

    def _train_backend(self, problem, seeds, cfg):
        self.require_available()
        return import_module(self.backend).train(problem, seeds, cfg)


class PPO(Algorithm):
    family = 'ppo'


class PPONN(PPO):
    name, variant, backend = 'ppo-nn', 'nn', 'algos.ppo_nn'

    def train(self, problem, seeds, cfg):
        return self._train_backend(problem, seeds, cfg)


class PPOLG(PPO):
    name, variant, backend = 'ppo-lg', 'lg', 'algos.ppo_lg'

    def train(self, problem, seeds, cfg):
        return self._train_backend(problem, seeds, cfg)


class TDMPC2(Algorithm):
    family = 'td-mpc2'


class TDMPC2LG(TDMPC2):
    # LG describes the actor prior; the model/critics are linear/quadratic.
    name, variant, backend = 'td-mpc2-lg', 'lg', 'algos.td_mpc2_lg'
    supported_problems = ('lqr_bounded',)

    def train(self, problem, seeds, cfg):
        return self._train_backend(problem, seeds, cfg)


class TDMPC2NN(TDMPC2):
    name, variant, backend = 'td-mpc2-nn', 'nn', 'algos.td_mpc2_nn'

    def train(self, problem, seeds, cfg):
        return self._train_backend(problem, seeds, cfg)


class MonteCarlo(Algorithm):
    family, name, backend = 'monte-carlo', 'monte_carlo', 'algos.monte_carlo'
    variant = 'tabular'

    def train(self, problem, seeds, cfg):
        return self._train_backend(problem, seeds, cfg)


VARIANTS = {cls.name: cls for cls in (PPONN, PPOLG, TDMPC2LG, TDMPC2NN, MonteCarlo)}
ALIASES = {'ppo': 'ppo-nn', 'ppo_linear': 'ppo-lg', 'ppo-lq': 'ppo-lg',
           'tdmpc2_lq': 'td-mpc2-lg', 'td-mpc2-lq': 'td-mpc2-lg'}


def canonical_name(name):
    return ALIASES.get(name, name)


def get_algorithm(name):
    name = canonical_name(name)
    if name not in VARIANTS:
        raise KeyError(f'Unknown algorithm {name!r}; known: {sorted(VARIANTS)}')
    return VARIANTS[name]()


def canonical_algorithms(names):
    names = tuple(canonical_name(name) for name in names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Select distinct algorithm variants; aliases cannot duplicate a variant')
    for name in names:
        get_algorithm(name).require_available()
    return names
