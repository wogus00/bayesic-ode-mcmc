import matplotlib.pyplot as plt
import numpy as np

def plot_trace(samples, parameter_names=None, title='Trace Plot'):
    """
    Plot trace plots for MCMC samples.

    Parameters:
      samples : numpy.ndarray
          MCMC samples with shape (num_samples, num_parameters).
      parameter_names : list of str, optional
          Names of the parameters for labeling.
      title : str, optional
          Title of the plot.
    """
    num_params = samples.shape[1]
    fig, axes = plt.subplots(num_params, 1, figsize=(8, 2*num_params), sharex=True)
    if num_params == 1:
        axes = [axes]
    for i in range(num_params):
        axes[i].plot(samples[:, i])
        axes[i].set_ylabel(parameter_names[i] if parameter_names else f'Param {i}')
    axes[-1].set_xlabel('Iteration')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_histogram(samples, parameter_names=None, title='Histogram'):
    """
    Plot histograms for MCMC samples.

    Parameters:
      samples : numpy.ndarray
          MCMC samples with shape (num_samples, num_parameters).
      parameter_names : list of str, optional
          Names of the parameters for labeling.
      title : str, optional
          Title of the plot.
    """
    num_params = samples.shape[1]
    fig, axes = plt.subplots(1, num_params, figsize=(4*num_params, 4))
    if num_params == 1:
        axes = [axes]
    for i in range(num_params):
        axes[i].hist(samples[:, i], bins=30, density=True, alpha=0.7)
        axes[i].set_xlabel(parameter_names[i] if parameter_names else f'Param {i}')
        axes[i].set_ylabel('Density')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_simulation(t, simulation_data, labels=None, title='Simulation'):
    """
    Plot simulation results.

    Parameters:
      t : array-like
          Time points.
      simulation_data : numpy.ndarray
          Simulation output (2D array with shape (len(t), num_series)).
      labels : list of str, optional
          Labels for each time series.
      title : str, optional
          Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    num_series = simulation_data.shape[1] if simulation_data.ndim > 1 else 1
    if num_series == 1:
        plt.plot(t, simulation_data)
    else:
        for i in range(num_series):
            plt.plot(t, simulation_data[:, i], label=labels[i] if labels else f'Series {i}')
        plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.tight_layout()
    plt.show()
