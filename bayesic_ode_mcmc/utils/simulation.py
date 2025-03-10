import numpy as np
from scipy.integrate import odeint

def add_noise(data, noise_std=1.0, random_seed=None):
    """
    Adds Gaussian noise to the given data.

    Parameters:
      data : array-like
          Original simulation data.
      noise_std : float, optional
          Standard deviation of the noise. Default is 1.0.
      random_seed : int, optional
          Seed for reproducibility.
      
    Returns:
      numpy.ndarray
          Data with added Gaussian noise.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = np.random.normal(scale=noise_std, size=np.array(data).shape)
    return np.array(data) + noise

def integrate_model(model_func, X0, t, args=(), rtol=1e-6, atol=1e-8):
    """
    Integrate a model given an ODE function.

    Parameters:
      model_func : function
          The function defining the ODE system.
      X0 : list or array
          Initial conditions.
      t : array-like
          Time points for integration.
      args : tuple, optional
          Extra arguments to pass to the ODE function.
      rtol : float, optional
          Relative tolerance for integration.
      atol : float, optional
          Absolute tolerance for integration.
      
    Returns:
      numpy.ndarray
          The integrated solution.
    """
    return odeint(model_func, X0, t, args=args, rtol=rtol, atol=atol)
