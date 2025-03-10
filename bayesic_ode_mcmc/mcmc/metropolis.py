import numpy as np

def metropolis_sampler(initial_theta, target_log_prob, num_samples, proposal_std):
    """
    Metropolis–Hastings sampler.
    
    Parameters:
      initial_theta : np.array
          Initial state.
      target_log_prob : function
          Function computing the log probability of the target distribution.
      num_samples : int
          Number of samples to generate.
      proposal_std : float or np.array
          Standard deviation(s) for the Gaussian proposal distribution.
          
    Returns:
      samples : np.array
          Array of samples with shape (num_samples, len(initial_theta)).
    """
    current_theta = np.array(initial_theta)
    samples = np.zeros((num_samples, len(current_theta)))
    
    current_log_prob = target_log_prob(current_theta)
    
    for i in range(num_samples):
        # Propose a new state from a Gaussian centered at current_theta
        proposal = np.random.normal(loc=current_theta, scale=proposal_std, size=current_theta.shape)
        proposal_log_prob = target_log_prob(proposal)
        
        # Calculate the log acceptance ratio
        log_accept_ratio = proposal_log_prob - current_log_prob
        
        if np.log(np.random.rand()) < log_accept_ratio:
            current_theta = proposal
            current_log_prob = proposal_log_prob
        
        samples[i, :] = current_theta
        
    return samples
