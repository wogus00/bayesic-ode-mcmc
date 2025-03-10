"""
abc package

This package contains functions and classes for performing Approximate Bayesian
Computation (ABC) that integrates simulation models with MCMC algorithms.
"""

from .abc_sampler import ABCSampler

__all__ = ['ABCSampler']
