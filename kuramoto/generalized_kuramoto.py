"""
Implements the N-dimensional (generalized) Kuramoto model.

Equation (2):
   d x_i / dt = Omega_i @ x_i
              + Proj_{x_i}(
                    c_i
                    + sum_j( J_{ij} @ x_j )
                )

where:
   - x_i in R^N is a unit vector for each oscillator i,
   - Omega_i is an N x N anti-symmetric matrix (natural frequency),
   - J_{ij} is an N x N coupling matrix from oscillator j to oscillator i,
   - c_i in R^N is a bias (“symmetry breaking”) vector for oscillator i,
   - Proj_{x_i}(y_i) = y_i - <y_i, x_i> x_i ensures updates remain tangent to x_i.

Author: Elder G. Veliz
Date:   2025-01-12
"""

import numpy as np
from scipy.integrate import odeint
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class KuramotoParams:
    """Parameters for the Generalized Kuramoto model."""
    Omega: np.ndarray  # natural frequencies
    J: np.ndarray      # coupling matrix
    c: np.ndarray      # bias vectors


class GeneralizedKuramoto:
    """
    Class for simulating a generalized (vector) Kuramoto model, where
    each oscillator is an N-dimensional unit vector.
    """

    def __init__(
        self,
        n_osc: int,
        dim:   int = 2,
        dt:    float = 0.01,
        T:     float = 10.0,
        Omega: Optional[np.ndarray] = None,
        J:     Optional[np.ndarray] = None,
        c:     Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        n_osc : int
            Number of oscillators.
        dim : int
            Dimension of each oscillator vector
            (e.g., 2 -> on the unit circle,
                   3 -> on the unit sphere,
                   etc.).
        dt : float
            Time step for integration.
        T : float
            Total simulation time.
        Omega : np.ndarray, optional
            Shape (n_osc, dim, dim).
            If None, defaults to zeros (no internal rotation).
            Must be anti-symmetric for “pure” rotations.
        J : np.ndarray, optional
            Shape (n_osc, n_osc, dim, dim). 
            Coupling from x_j to x_i
            If None, defaults to zeros.
        c : np.ndarray, optional
            Shape (n_osc, dim).
            External bias for each oscillator.
            If None, defaults to zeros.
        """
        self.n_osc = n_osc
        self.dim = dim
        self.dt = dt
        self.T = T

        self.params = self._initialize_parameters(Omega, J, c)

    def _initialize_parameters(
        self,
        Omega: Optional[np.ndarray],
        J: Optional[np.ndarray],
        c: Optional[np.ndarray]
    ) -> KuramotoParams:
        """Initialize model parameters."""
        if Omega is None:
            Omega = np.zeros((self.n_osc, self.dim, self.dim))
        else:
            self._validate_shape(Omega, (self.n_osc, self.dim, self.dim), "Omega")
            
        if J is None:
            J = np.zeros((self.n_osc, self.n_osc, self.dim, self.dim))
            # identity matrices for coupling
            eye_matrix = 0.1 * np.eye(self.dim)
            # broadcast to all pairs except self-coupling
            J[~np.eye(self.n_osc, dtype=bool)] = eye_matrix
        else:
            self._validate_shape(J, (self.n_osc, self.n_osc, self.dim, self.dim), "J")

        if c is None:
            c = np.zeros((self.n_osc, self.dim))
        else:
            self._validate_shape(c, (self.n_osc, self.dim), "c")

        return KuramotoParams(Omega=Omega, J=J, c=c)

    @staticmethod
    def _validate_shape(
        arr: np.ndarray,
        expected_shape: Tuple,
        name: str
    ) -> None:
        """Validate array shapes with descriptive error messages."""
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}. "
                f"Got {arr.shape}."
            )

    def derivative(
        self,
        y : np.ndarray,
        t : float
    ) -> np.ndarray:
        """
        Compute dy/dt at time t, flattened in y.

        Args:
            y : 1D numpy array
                Shape = (n_osc * dim,)
                Flattened states of all oscillators stacked.
                We'll reshape to (n_osc, dim) inside.
            t : float
               Current time. (Not necessarily used if system is time-invariant, but required by odeint).

        Returns:
            dydt : 1D numpy array 
                Shape (n_osc * dim,) 
                Represents derivative of all oscillators' states, flattened.
        """
        # 1) reshape y -> (n_osc, dim)
        x = y.reshape((self.n_osc, self.dim))

        # 2) natural frequency terms
        omega_terms = np.einsum('ijk,ik->ij', self.params.Omega, x)

        # 3) couple terms
        coupling_terms = np.einsum('ijkl,jl->ik', self.params.J, x)
        
        # 4) add bias terms
        total_input = coupling_terms + self.params.c

        # 5) project onto tangent spaces
        # compute dot products
        dots = np.sum(total_input * x, axis=1, keepdims=True)
        # subtract parallel components
        tangent_terms = total_input - dots * x

        # 6) combine terms
        dxdt = omega_terms + tangent_terms

        # 7) flatten dxdt
        return dxdt.flatten()

    def integrate(
        self,
        x0 : np.ndarray
    ):
        """
        Integrate the system from t=0 to t=T with initial condition x0.

        Args:
            x0: np.ndarray 
                Shape (n_osc, dim)
                The initial states for all oscillators. Must have norm=1 for each row ideally.

        Returns:
            times: 1D numpy array of shape (n_steps,)
                   The time array from 0 up to T.
            sol: 2D numpy array of shape (n_osc*dim, n_steps)
                 The flattened states of the system for each time step, i.e. node vs time.
        """
        n_steps = int(self.T / self.dt)
        times = np.linspace(0, self.T, n_steps)

        # shape: (n_osc * dim, n_steps)
        sol = odeint(self.derivative, x0.flatten(), times).T
        return times, sol

    @staticmethod
    def normalize_rows(
        mat : np.ndarray
    ) -> np.ndarray:
        """
        Normalizes each row of mat to unit length.
        mat: np.ndarray of shape (n_osc, dim)

        Returns:
            new_mat: same shape (n_osc, dim), 
                each row is normalized to have L2 norm 1.
        """
        norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
        return mat / norms

    def run(
        self,
        x0 : np.ndarray = None
    ):
        """
        Helper method to run a simulation from t=0 to t=T 
        using some initial condition x0.
        If x0 is None, random unit vectors are used.

        Args:
            x0: None or ndarray of shape (n_osc, dim), optional
                If None, random vectors are generated.

        Returns:
            times: shape (n_steps,)
            sol:   shape (n_osc*dim, n_steps)
        """
        if x0 is None:
            x0 = np.random.default_rng().normal(size=(self.n_osc, self.dim))
        else:
            self._validate_shape(x0, (self.n_osc, self.dim), "x0")

        x0 = self.normalize_rows(x0)
        return self.integrate(x0)

