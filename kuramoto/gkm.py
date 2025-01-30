import numpy as np
from scipy.integrate import odeint

class GeneralizedKuramoto:
    def __init__(
        self,
        n_osc,
        N=2,
        dt=0.01,
        T=10.0,
        Omega=None,
        c=None,
        J=None
    ):

        # store parameters
        self.n_osc = n_osc   # number of oscillators
        self.N = N           # dimension of each oscillator
        self.dt = dt         # integration time step
        self.T = T           # total integration time

        # if user does not provide Omega, default to zero (no intrinsic rotation)
        if Omega is None:
            # shape: (n_osc, N, N)
            self.Omega = np.zeros((n_osc, N, N), dtype=np.float32)
        else:
            # check shape
            assert Omega.shape == (n_osc, N, N), (
                "Omega must have shape (n_osc, N, N)"
            )
            self.Omega = Omega

        # if user does not provide c, default to zero vectors
        if c is None:
            # shape: (n_osc, N)
            self.c = np.zeros((n_osc, N), dtype=np.float32)
        else:
            # check shape
            assert c.shape == (n_osc, N), (
                "c must have shape (n_osc, N)"
            )
            self.c = c

        # if user does not provide J, default to zero matrix
        if J is None:
            # shape: (n_osc, n_osc)
            self.J = np.zeros((n_osc, n_osc, N, N), dtype=np.float32)
        else:
            # check shape
            assert J.shape == (n_osc, n_osc, N, N), (
                "J must have shape (n_osc, n_osc, N, N)"
            )
            self.J = J

        # Precompute flattened indices for sparse J (if applicable)
        self._precompute_sparse_indices()

    def _precompute_sparse_indices(self):
        """Cache indices of non-zero interactions for sparse J"""
        self.sparse_i, self.sparse_j = np.where(np.any(self.J != 0, axis=(2,3)))

    def init_x(self):
        """Vectorized initialization of oscillators"""
        x = np.random.normal(size=(self.n_osc, self.N)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    @staticmethod
    def _project_tangent(x_i, y):
        # compute dot product (x_i^T y)
        dot_prod = np.dot(x_i, y)
        # subtract off the component parallel to x_i
        proj = y - dot_prod * x_i
        return proj

    def _dxdt(self, x_flat, t):
        """Vectorized derivative calculation using Einstein summation"""
        # reshape flat state into (n_osc, N)
        x = x_flat.reshape((self.n_osc, self.N))

        # normalize x to ensure starting on sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms

        dxdt = np.zeros_like(x, dtype=np.float32)

        # Vectorized natural frequency terms
        omega_terms = np.einsum('ijk,ik->ij', self.Omega, x)
        
        # Vectorized coupling terms using sparse indices
        if hasattr(self, 'sparse_i'):
            J_flat = self.J[self.sparse_i, self.sparse_j]
            x_j = x[self.sparse_j]
            couplings = np.einsum('ijk,ik->ij', J_flat, x_j)
            np.add.at(dxdt, self.sparse_i, couplings)
        
        # Add forcing terms and project
        total_influence = self.c + dxdt

        # Project onto tangent space (orthogonal to x)
        projections = total_influence - np.einsum('ij,ij->i', total_influence, x)[:, None] * x
        
        return (omega_terms + projections).flatten()

    def integrate(self, x_init):
        # create a time array
        t_array = np.linspace(0, self.T, int(self.T / self.dt))

        # flatten x_init for odeint
        x0_flat = x_init.flatten()

        # integrate using odeint
        sol = odeint(self._dxdt, x0_flat, t_array)

        # reshape solution to (num_times, n_osc, N)
        x_timeseries = sol.reshape((len(t_array), self.n_osc, self.N))

        # normalize to ensure on unit sphere
        norms = np.linalg.norm(x_timeseries, axis=2, keepdims=True)
        x_timeseries = x_timeseries / norms

        return x_timeseries

    def run(self, x_init=None):
        # if no initial condition is given, use random unit vectors
        if x_init is None:
            x_init = self.init_x()

        # perform integration
        return self.integrate(x_init)