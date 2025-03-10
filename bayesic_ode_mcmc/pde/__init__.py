"""
pdes package

This package contains implementations of various PDE and ODE models,
including the Lotka–Volterra model and Reaction–Diffusion equations.
"""

from .lotka_volterra import simulate_lotka_volterra
from .reaction_diffusion import ReactionDiffusionSolver

__all__ = ['simulate_lotka_volterra', 'ReactionDiffusionSolver']
