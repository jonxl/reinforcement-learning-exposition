"""Execution-device selection; structured NumPy algorithms remain on CPU."""
import warnings
import torch

NEURAL_ALGORITHMS = {'ppo-nn', 'td-mpc2-nn'}


def select_devices(request, algorithm):
    if request not in ('cpu', 'gpu'):
        raise ValueError('device must be cpu or gpu')
    if request == 'cpu':
        return ['cpu']
    if algorithm not in NEURAL_ALGORITHMS:
        print(f'{algorithm}: CPU backend (GPU request applies to neural variants)', flush=True)
        return ['cpu']
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not count:
        warnings.warn('CUDA is unavailable; falling back to CPU.', RuntimeWarning, stacklevel=2)
        return ['cpu']
    return [f'cuda:{i}' for i in range(count)]
