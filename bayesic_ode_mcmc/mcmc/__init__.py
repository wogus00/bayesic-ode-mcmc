"""
mcmc package

This package contains implementations of various MCMC algorithms including
Hamiltonian Monte Carlo (HMC), Metropolis–Hastings, and Slice Sampling.
"""

from .hmc import hmc_sampler
from .metropolis import metropolis_sampler
from .slice_sampler import slice_sampler

__all__ = ['hmc_sampler', 'metropolis_sampler', 'slice_sampler']
