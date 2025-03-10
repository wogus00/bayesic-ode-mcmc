import numpy as np
from scipy.integrate import odeint

def lotka_volterra_ode(X, t, a, b, c, d):
    """
    Defines the Lotka–Volterra ODEs.
    
    Parameters:
      X : list or array
          [prey, predator]
      t : float
          Time variable (required by odeint)
      a, b, c, d : float
          Model parameters
          
    Returns:
      list
          Derivatives [d(prey)/dt, d(predator)/dt]
    """
    prey, predator = X
    dprey_dt = a * prey - b * prey * predator
    dpredator_dt = -c * predator + d * b * prey * predator
    return [dprey_dt, dpredator_dt]

def simulate_lotka_volterra(a, b, c=1.5, d=0.75, X0=None, t_end=15, num_points=100, noise_std=0.0, random_seed=None):
    """
    Simulate the Lotka–Volterra model.

    Parameters:
      a, b, c, d : float
          Model parameters.
      X0 : list, optional
          Initial condition [prey, predator]. Defaults to [10.0, 5.0].
      t_end : float, optional
          End time of simulation.
      num_points : int, optional
          Number of time points.
      noise_std : float, optional
          Standard deviation of Gaussian noise to add to the solution.
      random_seed : int, optional
          Seed for random number generator.

    Returns:
      t : numpy.ndarray
          Array of time points.
      sol : numpy.ndarray
          Simulated solution; shape (num_points, 2).
    """
    if X0 is None:
        X0 = [10.0, 5.0]
    t = np.linspace(0, t_end, num_points)
    sol = odeint(lotka_volterra_ode, X0, t, args=(a, b, c, d))
    
    if noise_std > 0.0:
        if random_seed is not None:
            np.random.seed(random_seed)
        noise = np.random.normal(scale=noise_std, size=sol.shape)
        sol = sol + noise
    
    return t, sol
