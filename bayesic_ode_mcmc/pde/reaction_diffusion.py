import numpy as np

class ReactionDiffusionSolver:
    def __init__(self, D, reaction_func, u0, dx, dt, t_end, boundary='Neumann'):
        """
        Initialize the Reaction–Diffusion solver.

        Parameters:
          D : float
              Diffusion coefficient.
          reaction_func : function
              Function f(u) that computes the reaction term given u.
          u0 : numpy.ndarray
              Initial condition (1D array) for the system.
          dx : float
              Spatial step size.
          dt : float
              Time step size.
          t_end : float
              End time for the simulation.
          boundary : str, optional
              Boundary condition type ('Neumann' for zero-gradient or 'Dirichlet' for fixed values).
        """
        self.D = D
        self.f = reaction_func
        self.u0 = u0.copy()
        self.dx = dx
        self.dt = dt
        self.t_end = t_end
        self.boundary = boundary
        
        self.N = len(u0)
        self.num_steps = int(t_end / dt)
        self.x = np.linspace(0, (self.N - 1) * dx, self.N)
    
    def step(self, u):
        """
        Perform one time step using an explicit finite difference scheme.
        
        Parameters:
          u : numpy.ndarray
              Current state.
        
        Returns:
          numpy.ndarray
              Updated state after one time step.
        """
        u_new = u.copy()
        # Update interior points with second-order central difference
        for i in range(1, self.N - 1):
            u_new[i] = u[i] + self.dt * (
                self.D * (u[i+1] - 2*u[i] + u[i-1]) / self.dx**2 + self.f(u[i])
            )
        # Apply boundary conditions
        if self.boundary == 'Neumann':
            # Zero-gradient (Neumann): approximate derivative as 0 at boundaries.
            u_new[0] = u[0] + self.dt * (
                self.D * (u[1] - u[0]) / self.dx**2 + self.f(u[0])
            )
            u_new[-1] = u[-1] + self.dt * (
                self.D * (u[-2] - u[-1]) / self.dx**2 + self.f(u[-1])
            )
        elif self.boundary == 'Dirichlet':
            # Fixed boundaries (Dirichlet): u[0] and u[-1] remain constant.
            u_new[0] = u[0]
            u_new[-1] = u[-1]
        else:
            raise ValueError("Unknown boundary condition type. Use 'Neumann' or 'Dirichlet'.")
        return u_new
    
    def solve(self):
        """
        Solve the reaction–diffusion equation over time.
        
        Returns:
          t : numpy.ndarray
              Time grid.
          x : numpy.ndarray
              Spatial grid.
          solution : numpy.ndarray
              Array of solution snapshots over time (shape: (num_steps+1, N)).
        """
        u = self.u0.copy()
        solution = [u.copy()]
        for step in range(self.num_steps):
            u = self.step(u)
            solution.append(u.copy())
        solution = np.array(solution)
        t = np.linspace(0, self.t_end, self.num_steps + 1)
        return t, self.x, solution
