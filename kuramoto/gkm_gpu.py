import cupy as cp
from cupyx.scipy.integrate import odeint
import numpy as np
from numba import jit
import opt_einsum as oe

class GPUKuramoto:
    def __init__(
        self,
        n_osc,
        N=2,
        dt=0.01,
        T=10.0,
        Omega=None,
        c=None,
        J=None,
        use_gpu=True
    ):
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        
        # store parameters
        self.n_osc = n_osc
        self.N = N
        self.dt = dt
        self.T = T

        # Convert arrays to GPU if needed
        if Omega is None:
            self.Omega = self.xp.zeros((n_osc, N, N), dtype=np.float32)
        else:
            self.Omega = self.xp.asarray(Omega, dtype=np.float32)

        if c is None:
            self.c = self.xp.zeros((n_osc, N), dtype=np.float32)
        else:
            self.c = self.xp.asarray(c, dtype=np.float32)

        if J is None:
            self.J = self.xp.zeros((n_osc, n_osc, N, N), dtype=np.float32)
        else:
            self.J = self.xp.asarray(J, dtype=np.float32)

        # Pre-optimize einsum paths
        self.omega_path = oe.contract_path('ijk,ik->ij', 
                                         self.Omega.shape, 
                                         (self.n_osc, self.N))[0]
        
        # Precompute sparse indices
        self._precompute_sparse_indices()

    def _precompute_sparse_indices(self):
        """Cache indices of non-zero interactions for sparse J"""
        if self.use_gpu:
            J_cpu = cp.asnumpy(self.J)
            self.sparse_i, self.sparse_j = np.where(np.any(J_cpu != 0, axis=(2,3)))
            self.sparse_i = cp.asarray(self.sparse_i)
            self.sparse_j = cp.asarray(self.sparse_j)
        else:
            self.sparse_i, self.sparse_j = np.where(np.any(self.J != 0, axis=(2,3)))

    def init_x(self):
        """Vectorized initialization of oscillators"""
        x = self.xp.random.normal(size=(self.n_osc, self.N)).astype(np.float32)
        return x / self.xp.linalg.norm(x, axis=1, keepdims=True)

    def _dxdt(self, x_flat, t):
        """Optimized derivative calculation"""
        x = x_flat.reshape((self.n_osc, self.N))
        
        # Normalize (maintaining precision)
        norms = self.xp.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        # Natural frequency terms (using pre-optimized path)
        omega_terms = oe.contract('ijk,ik->ij', self.Omega, x, optimize=self.omega_path)
        
        # Initialize output array
        dxdt = self.xp.zeros_like(x)
        
        # Coupling terms using sparse indices
        if hasattr(self, 'sparse_i'):
            J_flat = self.J[self.sparse_i, self.sparse_j]
            x_j = x[self.sparse_j]
            couplings = oe.contract('ijk,ik->ij', J_flat, x_j)
            # Use scatter_add for GPU compatibility
            if self.use_gpu:
                dxdt = cp.scatter_add(dxdt, self.sparse_i, couplings)
            else:
                np.add.at(dxdt, self.sparse_i, couplings)
        
        # Add forcing terms
        total_influence = self.c + dxdt
        
        # Project onto tangent space (optimized)
        dots = self.xp.sum(total_influence * x, axis=1, keepdims=True)
        projections = total_influence - dots * x
        
        result = (omega_terms + projections)
        
        return result.flatten()

    def integrate(self, x_init):
        # Create time array
        t_array = self.xp.linspace(0, self.T, int(self.T / self.dt))
        
        # Convert initial conditions if needed
        if self.use_gpu and not isinstance(x_init, cp.ndarray):
            x_init = cp.asarray(x_init)
        
        # Flatten for integration
        x0_flat = x_init.flatten()
        
        # Integrate
        sol = odeint(self._dxdt, x0_flat, t_array)
        
        # Reshape and normalize
        x_timeseries = sol.reshape((len(t_array), self.n_osc, self.N))
        norms = self.xp.linalg.norm(x_timeseries, axis=2, keepdims=True)
        x_timeseries = x_timeseries / norms
        
        # Convert back to CPU if needed
        if self.use_gpu:
            x_timeseries = cp.asnumpy(x_timeseries)
        
        return x_timeseries

    def run(self, x_init=None):
        if x_init is None:
            x_init = self.init_x()
        return self.integrate(x_init)