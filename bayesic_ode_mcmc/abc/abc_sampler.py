import numpy as np

class ABCSampler:
    """
    A generic Approximate Bayesian Computation (ABC) sampler that uses MCMC.
    
    This class ties together a simulation model, a distance metric, a prior,
    and an MCMC sampler to perform likelihood-free inference.
    
    Parameters:
      simulate_func : function
          Function to simulate data given parameters. It should accept parameters 
          (as separate arguments or as an array unpacked) and return simulated data.
      distance_func : function
          Function that computes the distance between simulated and observed data.
      prior : function
          Function that evaluates the prior probability (or indicator function) for 
          parameters. It should return a non-zero value if the parameters are in 
          the prior support, and zero otherwise.
      mcmc_sampler : function
          An MCMC sampler function (e.g., metropolis_sampler, hmc_sampler, or slice_sampler)
          that accepts at least the following arguments: initial_theta, target_log_prob,
          num_samples, and any additional keyword arguments.
      observed_data : array-like
          The observed data to compare against the simulated data.
      epsilon : float
          Tolerance (or scaling) parameter used in the ABC kernel.
    """
    def __init__(self, simulate_func, distance_func, prior, mcmc_sampler, observed_data, epsilon):
        self.simulate_func = simulate_func
        self.distance_func = distance_func
        self.prior = prior
        self.mcmc_sampler = mcmc_sampler
        self.observed_data = observed_data
        self.epsilon = epsilon
        
    def target_log_prob(self, theta):
        """
        Define an approximate log probability function for ABC.
        
        This function uses a Gaussian kernel based on the distance between
        simulated and observed data. If the parameters are not in the prior support,
        it returns -np.inf.
        
        Parameters:
          theta : array-like
              The parameter vector.
              
        Returns:
          float
              The log probability.
        """
        # Check prior support
        if self.prior(theta) == 0:
            return -np.inf
        
        # Simulate data using the provided simulation function
        simulated = self.simulate_func(*theta)
        
        # Compute the distance between simulated and observed data
        d = self.distance_func(simulated, self.observed_data)
        
        # Gaussian kernel: exp(-d^2/(2*epsilon^2))
        likelihood = np.exp(-d**2 / (2 * self.epsilon**2))
        return np.log(likelihood)
    
    def sample(self, initial_theta, num_samples, **mcmc_kwargs):
        """
        Generate samples using the specified MCMC sampler.
        
        Parameters:
          initial_theta : array-like
              The starting parameter values.
          num_samples : int
              Number of samples to generate.
          **mcmc_kwargs :
              Additional keyword arguments to pass to the MCMC sampler.
              
        Returns:
          samples : np.array
              Array of MCMC samples of parameters.
        """
        # Use the provided MCMC sampler with our target_log_prob as the target.
        return self.mcmc_sampler(initial_theta, self.target_log_prob, num_samples, **mcmc_kwargs)
