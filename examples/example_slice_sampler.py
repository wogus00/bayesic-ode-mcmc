#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from bayesic_ode_mcmc.pde.lotka_volterra import simulate_lotka_volterra
from bayesic_ode_mcmc.pde.reaction_diffusion import ReactionDiffusionSolver
from bayesic_ode_mcmc.abc.abc_sampler import ABCSampler
from bayesic_ode_mcmc.mcmc.slice_sampler import slice_sampler
from bayesic_ode_mcmc.utils.diagnostics import plot_trace, plot_histogram

def distance_func(simulated, observed):
    return np.linalg.norm(np.array(simulated) - np.array(observed))

def lotka_volterra_prior(theta):
    a, b = theta
    return 1 if (0 < a < 2 and 0 < b < 0.5) else 0

def reaction_diffusion_prior(theta):
    D = theta[0]
    return 1 if (0.1 < D < 2.0) else 0

# ==========================
# Lotka–Volterra Example with Slice Sampling
# ==========================
print("Lotka–Volterra model with Slice Sampling")

true_params_lv = [1.0, 0.1]
t_lv, observed_lv = simulate_lotka_volterra(*true_params_lv, noise_std=0.5, random_seed=42)

def simulate_lv(a, b):
    t, sim = simulate_lotka_volterra(a, b, noise_std=0.0)
    return sim

epsilon_lv = 15.0
abc_lv = ABCSampler(simulate_func=simulate_lv,
                    distance_func=distance_func,
                    prior=lotka_volterra_prior,
                    mcmc_sampler=slice_sampler,
                    observed_data=observed_lv,
                    epsilon=epsilon_lv)

initial_theta_lv = [0.8, 0.15]
num_samples = 1000
samples_lv = abc_lv.sample(initial_theta=initial_theta_lv, num_samples=num_samples,
                           step_out=True, w=0.1, m=100)

plot_trace(samples_lv, parameter_names=['a', 'b'], title='Lotka–Volterra Slice Trace')
plot_histogram(samples_lv, parameter_names=['a', 'b'], title='Lotka–Volterra Slice Histogram')

# ==========================
# Reaction–Diffusion Example with Slice Sampling
# ==========================
print("Reaction–Diffusion model with Slice Sampling")

def reaction_func(u):
    return u - u**3

true_D = 0.5
N = 50
u0 = np.sin(np.linspace(0, np.pi, N))
dx = 0.1
dt = 0.001
t_end = 1.0

solver = ReactionDiffusionSolver(D=true_D, reaction_func=reaction_func, u0=u0, dx=dx, dt=dt, t_end=t_end)
t_rd, x_rd, sim_rd = solver.solve()
observed_rd = sim_rd + np.random.normal(scale=0.1, size=sim_rd.shape)

def simulate_rd(D):
    solver = ReactionDiffusionSolver(D=D, reaction_func=reaction_func, u0=u0, dx=dx, dt=dt, t_end=t_end)
    t_sim, x_sim, sim = solver.solve()
    return sim

epsilon_rd = 5.0
abc_rd = ABCSampler(simulate_func=simulate_rd,
                    distance_func=distance_func,
                    prior=reaction_diffusion_prior,
                    mcmc_sampler=slice_sampler,
                    observed_data=observed_rd,
                    epsilon=epsilon_rd)

initial_theta_rd = [0.7]
samples_rd = abc_rd.sample(initial_theta=initial_theta_rd, num_samples=num_samples,
                           step_out=True, w=0.1, m=100)

plot_trace(samples_rd, parameter_names=['D'], title='Reaction–Diffusion Slice Trace')
plot_histogram(samples_rd, parameter_names=['D'], title='Reaction–Diffusion Slice Histogram')
