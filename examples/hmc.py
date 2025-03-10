#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from bayesic_ode_mcmc.pde.lotka_volterra import simulate_lotka_volterra
from bayesic_ode_mcmc.pde.reaction_diffusion import ReactionDiffusionSolver
from bayesic_ode_mcmc.abc.abc_sampler import ABCSampler
from bayesic_ode_mcmc.mcmc.hmc import hmc_sampler
from bayesic_ode_mcmc.utils.diagnostics import plot_trace, plot_histogram

# Finite-difference gradient approximation.
def finite_diff_grad(f, theta, eps=1e-5):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = np.array(theta, dtype=float)
        theta_minus = np.array(theta, dtype=float)
        theta_plus[i] += eps
        theta_minus[i] -= eps
        grad[i] = (f(theta_plus) - f(theta_minus)) / (2 * eps)
    return grad

def distance_func(simulated, observed):
    return np.linalg.norm(np.array(simulated) - np.array(observed))

def lotka_volterra_prior(theta):
    a, b = theta
    return 1 if (0 < a < 2 and 0 < b < 0.5) else 0

def reaction_diffusion_prior(theta):
    D = theta[0]
    return 1 if (0.1 < D < 2.0) else 0

# ==========================
# Lotka–Volterra Example with HMC
# ==========================
print("Lotka–Volterra model with HMC")

true_params_lv = [1.0, 0.1]
t_lv, observed_lv = simulate_lotka_volterra(*true_params_lv, noise_std=0.5, random_seed=42)

def simulate_lv(a, b):
    t, sim = simulate_lotka_volterra(a, b, noise_std=0.0)
    return sim

epsilon_lv = 15.0
abc_lv = ABCSampler(simulate_func=simulate_lv,
                    distance_func=distance_func,
                    prior=lotka_volterra_prior,
                    mcmc_sampler=hmc_sampler,
                    observed_data=observed_lv,
                    epsilon=epsilon_lv)

# Define gradient function on the ABC target log probability.
grad_log_prob_lv = lambda theta: finite_diff_grad(abc_lv.target_log_prob, theta)

initial_theta_lv = [0.8, 0.15]
num_samples = 1000
samples_lv = abc_lv.sample(initial_theta=initial_theta_lv, num_samples=num_samples,
                           step_size=0.01, num_steps=10, mass=1.0, grad_log_prob=grad_log_prob_lv)

plot_trace(samples_lv, parameter_names=['a', 'b'], title='Lotka–Volterra HMC Trace')
plot_histogram(samples_lv, parameter_names=['a', 'b'], title='Lotka–Volterra HMC Histogram')

# ==========================
# Reaction–Diffusion Example with HMC
# ==========================
print("Reaction–Diffusion model with HMC")

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
                    mcmc_sampler=hmc_sampler,
                    observed_data=observed_rd,
                    epsilon=epsilon_rd)

grad_log_prob_rd = lambda theta: finite_diff_grad(abc_rd.target_log_prob, theta)

initial_theta_rd = [0.7]
samples_rd = abc_rd.sample(initial_theta=initial_theta_rd, num_samples=num_samples,
                           step_size=0.01, num_steps=10, mass=1.0, grad_log_prob=grad_log_prob_rd)

plot_trace(samples_rd, parameter_names=['D'], title='Reaction–Diffusion HMC Trace')
plot_histogram(samples_rd, parameter_names=['D'], title='Reaction–Diffusion HMC Histogram')
