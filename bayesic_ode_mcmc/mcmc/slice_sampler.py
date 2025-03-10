import numpy as np

def slice_sampler(initial_theta, target_log_prob, num_samples, step_out=True, w=1.0, m=100):
    """
    Slice sampler using coordinate-wise sampling.
    
    Parameters:
      initial_theta : np.array
          Initial state.
      target_log_prob : function
          Function computing the log probability of the target distribution.
      num_samples : int
          Number of samples to generate.
      step_out : bool, optional
          Whether to perform a step-out procedure to find the slice interval. Default is True.
      w : float, optional
          Initial bracket width. Default is 1.0.
      m : int, optional
          Maximum number of step-out steps. Default is 100.
          
    Returns:
      samples : np.array
          Array of samples with shape (num_samples, len(initial_theta)).
    """
    current_theta = np.array(initial_theta)
    dim = len(current_theta)
    samples = np.zeros((num_samples, dim))
    
    for i in range(num_samples):
        for d in range(dim):
            # Current log probability
            current_log_prob = target_log_prob(current_theta)
            # Draw a vertical level uniformly (in log-space)
            log_u = current_log_prob + np.log(np.random.rand())
            
            # Initialize the slice interval [L, R]
            L = current_theta[d] - w * np.random.rand()
            R = L + w
            
            # Step-out procedure to find the interval
            if step_out:
                j = 0
                while j < m and target_log_prob(set_value(current_theta, d, L)) > log_u:
                    L -= w
                    j += 1
                j = 0
                while j < m and target_log_prob(set_value(current_theta, d, R)) > log_u:
                    R += w
                    j += 1
            
            # Sample a new value for coordinate d
            new_val = np.random.uniform(L, R)
            while target_log_prob(set_value(current_theta, d, new_val)) < log_u:
                # Shrink the interval if proposal is rejected
                if new_val < current_theta[d]:
                    L = new_val
                else:
                    R = new_val
                new_val = np.random.uniform(L, R)
            
            current_theta[d] = new_val
        
        samples[i, :] = current_theta
        
    return samples

def set_value(theta, index, value):
    """
    Utility function to create a new array with theta[index] set to value.
    """
    theta_new = np.array(theta)
    theta_new[index] = value
    return theta_new
