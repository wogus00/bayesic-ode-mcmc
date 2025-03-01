# **bayes-ode-mcmc**  
*A Python package for Bayesian inference on ODE systems using MCMC sampling algorithms*  

## **Overview**  
`bayes-ode-mcmc` is a Python package that performs **Bayesian parameter estimation** for **Ordinary Differential Equation (ODE) systems** using **Markov Chain Monte Carlo (MCMC) sampling**. The package provides an easy-to-use interface for defining ODE models, specifying priors, and applying **Slice Sampling** and **Metropolis-Hastings (MH) algorithms** to infer unknown parameters.  

## **Features**  
✅ Define custom ODE models using Python functions  
✅ Perform Bayesian inference with **Slice Sampler** and **Metropolis-Hastings** algorithms  
✅ Supports **custom priors and likelihood functions**  
✅ Utilizes **numerical solvers** for ODE integration (`scipy.integrate.solve_ivp`)  
✅ Provides **diagnostic tools** (trace plots, effective sample size, and convergence checks)  
✅ Designed for flexibility, enabling easy integration with real-world datasets  

## **Installation**  
Install the package using pip:  
```bash
pip install bayes-ode-mcmc
