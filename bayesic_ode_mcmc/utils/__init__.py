"""
utils package

This package contains utility functions for running simulations and performing diagnostics.
It includes helper functions for noise addition, ODE integration, and plotting routines.
"""

from .simulation import add_noise, integrate_model
from .diagnostics import plot_trace, plot_histogram, plot_simulation

__all__ = ['add_noise', 'integrate_model', 'plot_trace', 'plot_histogram', 'plot_simulation']
