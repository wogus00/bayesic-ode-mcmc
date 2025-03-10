import numpy as np

def hmc_sampler(initial_theta, target_log_prob, grad_log_prob, num_samples, step_size, num_steps, mass=1.0):
    """
    Hamiltonian Monte Carlo (HMC) sampler.
    
    Parameters:
      initial_theta : np.array
          Initial state.
      target_log_prob : function
          Function computing the log probability of the target distribution.
      grad_log_prob : function
          Function computing the gradient of the log probability.
      num_samples : int
          Number of samples to generate.
      step_size : float
          Step size for the leapfrog integration.
      num_steps : int
          Number of leapfrog steps per iteration.
      mass : float or np.array, optional
          Mass (or mass matrix) for the momentum. Default is 1.0.
          
    Returns:
      samples : np.array
          Array of sampled states with shape (num_samples, len(initial_theta)).
    """
    samples = np.zeros((num_samples, len(initial_theta)))
    current_theta = np.array(initial_theta)
    
    for i in range(num_samples):
        # Sample momentum from N(0, mass)
        current_p = np.random.normal(loc=0, scale=np.sqrt(mass), size=current_theta.shape)
        
        # Store the current state for the leapfrog integration
        theta_proposal = current_theta.copy()
        p_proposal = current_p.copy()
        
        # Calculate the current Hamiltonian
        current_U = -target_log_prob(current_theta)
        current_K = np.sum(current_p**2) / (2 * mass)
        
        # Perform leapfrog integration
        # Half-step update for momentum
        p_proposal -= 0.5 * step_size * (-grad_log_prob(theta_proposal))
        for _ in range(num_steps):
            # Full-step update for position
            theta_proposal += step_size * p_proposal / mass
            # Full-step update for momentum except at the last step
            if _ != num_steps - 1:
                p_proposal -= step_size * (-grad_log_prob(theta_proposal))
        # Final half-step update for momentum
        p_proposal -= 0.5 * step_size * (-grad_log_prob(theta_proposal))
        
        # Negate momentum to make proposal symmetric
        p_proposal = -p_proposal
        
        # Calculate the proposed Hamiltonian
        proposed_U = -target_log_prob(theta_proposal)
        proposed_K = np.sum(p_proposal**2) / (2 * mass)
        
        # Accept/reject step
        acceptance_prob = np.exp((current_U + current_K) - (proposed_U + proposed_K))
        if np.random.rand() < acceptance_prob:
            current_theta = theta_proposal  # accept
        
        samples[i, :] = current_theta
        
    return samples
