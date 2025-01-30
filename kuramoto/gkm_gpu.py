import numpy as np
import torch
from scipy.integrate import odeint

class FlexibleKuramoto:
    def __init__(
        self,
        n_osc,
        N=2,
        dt=0.01,
        T=10.0,
        Omega=None,
        c=None,
        J=None,
        device='mps'  # Use Metal Performance Shaders by default
    ):
        # Check if MPS is available
        self.device = (torch.device(device) 
                      if torch.backends.mps.is_available() and device == 'mps'
                      else torch.device('cpu'))
        
        # Store parameters
        self.n_osc = n_osc
        self.N = N
        self.dt = dt
        self.T = T

        # Convert numpy arrays to torch tensors and move to appropriate device
        if Omega is None:
            self.Omega = torch.zeros((n_osc, N, N), dtype=torch.float32, device=self.device)
        else:
            self.Omega = torch.tensor(Omega, dtype=torch.float32, device=self.device)

        if c is None:
            self.c = torch.zeros((n_osc, N), dtype=torch.float32, device=self.device)
        else:
            self.c = torch.tensor(c, dtype=torch.float32, device=self.device)

        if J is None:
            self.J = torch.zeros((n_osc, n_osc, N, N), dtype=torch.float32, device=self.device)
        else:
            self.J = torch.tensor(J, dtype=torch.float32, device=self.device)

        # Precompute sparse indices
        self._precompute_sparse_indices()

    def _precompute_sparse_indices(self):
        """Cache indices of non-zero interactions for sparse J"""
        # Move J to CPU temporarily for numpy operations
        J_cpu = self.J.cpu().numpy()
        self.sparse_i, self.sparse_j = np.where(np.any(J_cpu != 0, axis=(2,3)))
        # Convert indices to torch tensors
        self.sparse_i = torch.tensor(self.sparse_i, device=self.device)
        self.sparse_j = torch.tensor(self.sparse_j, device=self.device)

    def init_x(self):
        """Vectorized initialization of oscillators"""
        x = torch.randn(self.n_osc, self.N, dtype=torch.float32, device=self.device)
        return x / torch.norm(x, dim=1, keepdim=True)

    @staticmethod
    def _project_tangent(x_i, y):
        # Compute projection using torch operations
        dot_prod = torch.dot(x_i, y)
        proj = y - dot_prod * x_i
        return proj

    def _dxdt(self, x_flat, t):
        """Vectorized derivative calculation optimized for M1"""
        # Reshape flat state into (n_osc, N)
        x = torch.tensor(x_flat, dtype=torch.float32, device=self.device).reshape(self.n_osc, self.N)
        
        # Normalize x to ensure starting on sphere
        x = x / torch.norm(x, dim=1, keepdim=True)

        # Initialize output tensor
        dxdt = torch.zeros_like(x, device=self.device)

        # Compute natural frequency terms using torch.einsum
        omega_terms = torch.einsum('ijk,ik->ij', self.Omega, x)
        
        # Compute coupling terms using sparse indices
        if hasattr(self, 'sparse_i'):
            J_flat = self.J[self.sparse_i, self.sparse_j]
            x_j = x[self.sparse_j]
            couplings = torch.einsum('ijk,ik->ij', J_flat, x_j)
            # Use scatter_add_ for efficient sparse updates
            dxdt.scatter_add_(0, self.sparse_i.unsqueeze(1).expand(-1, self.N), couplings)
        
        # Add forcing terms
        total_influence = self.c + dxdt

        # Project onto tangent space using efficient batch operations
        x_dot_influence = torch.sum(total_influence * x, dim=1, keepdim=True)
        projections = total_influence - x_dot_influence * x
        
        # Combine terms and move back to CPU for scipy integration
        result = (omega_terms + projections).cpu().numpy()
        return result.flatten()

    def integrate(self, x_init):
        # Create time array
        t_array = np.linspace(0, self.T, int(self.T / self.dt))

        # Flatten initial conditions
        x0_flat = x_init.cpu().numpy().flatten()

        # Integrate using scipy's odeint
        sol = odeint(self._dxdt, x0_flat, t_array)

        # Reshape solution and convert to torch tensor
        x_timeseries = torch.tensor(
            sol.reshape(len(t_array), self.n_osc, self.N),
            dtype=torch.float32,
            device=self.device
        )

        # Normalize to ensure on unit sphere
        x_timeseries = x_timeseries / torch.norm(x_timeseries, dim=2, keepdim=True)

        return x_timeseries

    def run(self, x_init=None):
        if x_init is None:
            x_init = self.init_x()
        return self.integrate(x_init)