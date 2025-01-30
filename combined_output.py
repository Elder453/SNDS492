import cupy as cp
from cupyx.scipy.integrate import odeint
import numpy as np
from numba import jit
import opt_einsum as oe

class OptimizedKuramoto:
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
        return self.integrate(x_init)#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from PIL import Image

from kuramoto import GeneralizedKuramoto


# In[2]:


# ----------------------------------------------------------------------
# Step 1: Define Problem Size and Load Fish Mask
# ----------------------------------------------------------------------
H, W = 64, 64            # height and width of the oscillator grid
n_osc = H * W            # total number of oscillators
N = 3                    # dimension of each oscillator's state vector

# Load the fish mask image
mask_path = "../images/fish_mask.png"
mask_img = Image.open(mask_path).convert("L") # convert to grayscale
mask_img = mask_img.resize((W, H))            # ensure the image is 64x64
fish_mask = np.array(mask_img) / 255.0        # normalize pixel values to [0,1]

# ----------------------------------------------------------------------
# Step 2: Create Forcing Vectors (c)
# ----------------------------------------------------------------------
# Initialize c as zeros
c = np.zeros((n_osc, N), dtype=np.float32)
fish_mask_flat = fish_mask.reshape(-1)
# Set forcing direction to [1,1,1] for oscillators inside the fish
c[fish_mask_flat > 0.5] = 1.0 / np.sqrt(3)  # Normalize to unit vector

# ----------------------------------------------------------------------
# Step 3: Build Adjacency Matrix (J) Using Gaussian Kernel
# ----------------------------------------------------------------------
kernel_size = 9
sigma = 3.0

# Create kernel with continuous coordinates (center at 4.5, 4.5)
kernel = np.zeros((kernel_size, kernel_size), dtype=float)
for kr in range(kernel_size):
    for kc in range(kernel_size):
        # Distance from center (4.5, 4.5)
        dx = kr - 4.5
        dy = kc - 4.5
        dist2 = dx**2 + dy**2
        kernel[kr, kc] = np.exp(-dist2 / (2 * sigma**2))
kernel /= kernel.sum()  # Normalize so sum of weights = 1

# Create 3D rotational coupling matrix
def get_3d_rotation_matrix(theta=0.5):
    """Generate a 3D rotation matrix that rotates around all axes"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ], dtype=np.float32)
    
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float32)
    
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Combine rotations
    return Rx @ Ry @ Rz

# Rotational coupling matrix (N=2)
base_matrix = get_3d_rotation_matrix()

# Initialize J as 4D tensor: (n_osc, n_osc, N, N)
J = np.zeros((n_osc, n_osc, N, N), dtype=np.float32)

# Populate J with Gaussian-scaled rotational couplings
for row in range(H):
    for col in range(W):
        i = row * W + col  # Current oscillator index
        for kr in range(kernel_size):
            for kc in range(kernel_size):
                # Integer offsets from oscillator (row, col)
                dr = kr - 4  # Maps [0, 8] → [-4, +4]
                dc = kc - 4
                rr = row + dr
                cc2 = col + dc
                if 0 <= rr < H and 0 <= cc2 < W:
                    j = rr * W + cc2  # Neighbor oscillator index
                    weight = kernel[kr, kc]
                    J[i, j] = weight * base_matrix  # (2, 2) matrix

# ----------------------------------------------------------------------
# Step 4: Define Natural Frequency Matrix (Omega)
#         Here, we set Omega to zero matrices (no intrinsic rotation)
# ----------------------------------------------------------------------

# Define a base anti-symmetric matrix (natural frequency)
omega_val = 0.5  # Adjust this to control rotation speed
def get_3d_skew_symmetric():
    """Generate a random 3D skew-symmetric matrix for natural frequencies"""
    A = np.random.randn(3, 3) * omega_val
    return (A - A.T) / 2

# Initialize Omega with spatially varying natural frequencies
Omega = np.zeros((n_osc, N, N), dtype=np.float32) # shape: (4096, 2, 2)
base_omega = get_3d_skew_symmetric()
for i in range(n_osc):
    # Add randomness to natural frequencies (optional)
    Omega[i] = base_omega * (1 + 0.01 * np.random.randn())


# In[28]:


# ----------------------------------------------------------------------
# Step 5: Initialize the GeneralizedKuramoto Model
# ----------------------------------------------------------------------
gamma = 1    # step size for updates
T = 250      # total number of discrete steps

# Instantiate the GeneralizedKuramoto class
gk = GeneralizedKuramoto(
    n_osc=n_osc,
    N=N,
    dt=gamma,       # step size corresponds to gamma in the discrete update
    T=T * gamma,    # total time is number of steps times step size
    Omega=Omega,
    c=c,
    J=J
)

# ----------------------------------------------------------------------
# Step 6: Initialize Oscillator States (x0)
# ----------------------------------------------------------------------
x0 = gk.init_x()  # shape: (4096, 2)

# ----------------------------------------------------------------------
# Step 7: Run the Simulation
# ----------------------------------------------------------------------
# Integrate the ODE using the GeneralizedKuramoto class
x_timeseries = gk.run(x_init=x0)  # shape: (num_times, 4096, 2)


# In[29]:


# ----------------------------------------------------------------------
# Step 8: Compute Energy E(t) Over Time
# ----------------------------------------------------------------------
def compute_energy(x_series, J, c):
    """Compute energy E(t) for 3D oscillators"""
    num_times = x_series.shape[0]
    E = np.zeros(num_times)
    for t in range(num_times):
        X = x_series[t]  # shape: (n_osc, N)
        # Compute -sum_{i,j} x_i^T @ J_ij @ x_j using einsum
        coupling_energy = -np.einsum('ia,ijab,jb->', X, J, X)
        # Compute -sum_i c_i^T x_i
        forcing_energy = -np.sum(c * X)
        E[t] = coupling_energy + forcing_energy
    return E

# Calculate energy
E = compute_energy(x_timeseries, J, c)


# In[30]:


# ----------------------------------------------------------------------
# Step 9: Two-row Visualization with Mask and Evolution
# ----------------------------------------------------------------------
def create_3d_animation_with_energy(x_timeseries, fish_mask, E, H, W, interval=50):
    """Create an animation of 3D oscillator states with energy plot"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Create three subplots: mask, 3D scatter, and energy plot
    ax_mask = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
    ax_energy = fig.add_subplot(gs[1, :])
    
    # Show the mask
    ax_mask.imshow(fish_mask, cmap='gray')
    ax_mask.set_title('Mask')
    ax_mask.axis('off')
    
    # Initialize 3D scatter plot
    fish_points = np.where(fish_mask.flatten() > 0.5)[0]
    x_fish = x_timeseries[:, fish_points, :]  # Only plot points inside fish
    
    scatter = ax_3d.scatter(
        x_fish[0, :, 0],
        x_fish[0, :, 1],
        x_fish[0, :, 2],
        c=np.arange(len(fish_points)),
        cmap='viridis',
        alpha=0.6
    )
    
    # Set equal aspect ratio for 3D plot
    ax_3d.set_box_aspect([1,1,1])
    ax_3d.set_xlim([-1.1, 1.1])
    ax_3d.set_ylim([-1.1, 1.1])
    ax_3d.set_zlim([-1.1, 1.1])
    
    # Add unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax_3d.plot_surface(x, y, z, color='gray', alpha=0.1)
    
    # Initialize energy plot
    time_points = np.arange(len(E)) * gamma
    line, = ax_energy.plot(time_points[0:1], E[0:1], 'b-')
    ax_energy.set_xlabel('Time')
    ax_energy.set_ylabel('Energy E(t)')
    ax_energy.set_xlim([0, time_points[-1]])
    ax_energy.set_ylim([np.min(E), np.max(E)])
    ax_energy.grid(True)
    
    # Add vertical line for current time
    vline = ax_energy.axvline(x=0, color='r', linestyle='--')
    
    # Animation update function
    def update(frame):
        # Update scatter plot data
        scatter._offsets3d = (
            x_fish[frame, :, 0],
            x_fish[frame, :, 1],
            x_fish[frame, :, 2]
        )
        ax_3d.view_init(elev=30, azim=frame/2)  # Rotate view
        
        # Update energy plot
        line.set_data(time_points[:frame+1], E[:frame+1])
        vline.set_xdata([time_points[frame], time_points[frame]])
        
        return scatter, line, vline
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(x_timeseries),
        interval=interval, blit=True
    )
    
    plt.tight_layout()
    plt.close()
    return anim

# Create and display animation with energy
anim = create_3d_animation_with_energy(x_timeseries, fish_mask, E, H, W, interval=50)

# Save animation as HTML5 video
from IPython.display import HTML
HTML(anim.to_jshtml())

# Also create a static plot of the energy evolution
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(E)) * gamma, E, 'b-')
plt.xlabel('Time')
plt.ylabel('Energy E(t)')
plt.title('Energy Evolution in 3D Kuramoto Model')
plt.grid(True)
plt.show()


# In[31]:


import matplotlib.animation as animation
from IPython.display import HTML

def create_oscillator_animation(x_timeseries, fish_mask, H, W, gamma, interval=50, save_gif=False, filename='oscillator.gif'):
    """
    Create an animation of oscillator states evolving over time.
    
    Parameters:
    - x_timeseries: ndarray, shape (num_times, n_osc, 2) containing oscillator states
    - fish_mask: ndarray, shape (H, W) containing the binary mask
    - H, W: integers, height and width of the grid
    - gamma: float, time step size
    - interval: int, delay between frames in milliseconds
    - save_gif: bool, whether to save the animation as a GIF
    - filename: str, name of the output GIF file if saving
    
    Returns:
    - HTML object containing the animation for Jupyter notebook display
    """
    # Create figure with two subplots side by side
    fig, (ax_mask, ax_state) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display the mask in the left subplot
    ax_mask.imshow(fish_mask, cmap='gray')
    ax_mask.set_title('Initial Mask')
    ax_mask.axis('off')
    
    # Initialize the oscillator state plot
    initial_state = map_states_to_rgb(x_timeseries[0], H, W)
    im = ax_state.imshow(initial_state, interpolation='nearest')
    #ax_state.set_title(f't = 0.00')
    ax_state.axis('off')
    
    # Text for displaying the time
    time_text = ax_state.text(0.02, 1.05, '', transform=ax_state.transAxes)
    
    def update(frame):
        """Update function for animation"""
        # Convert oscillator state to RGB
        rgb = map_states_to_rgb(x_timeseries[frame], H, W)
        
        # Update the image
        im.set_array(rgb)
        
        # Update the time display
        current_time = frame * gamma
        time_text.set_text(f't = {current_time:.2f}')
        
        return [im, time_text]
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(x_timeseries),
        interval=interval,
        blit=True
    )
    
    # Save as GIF if requested
    if save_gif:
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='pillow')
        print("Done saving!")
    
    plt.close()  # Close the figure to free memory
    
    # Return the animation as HTML for display in notebook
    return HTML(anim.to_jshtml())

# Create and display the animation
animation_html = create_oscillator_animation(
    x_timeseries=x_timeseries,
    fish_mask=fish_mask,
    H=H,
    W=W,
    gamma=gamma,
    interval=50,  # 50ms between frames
    save_gif=True,  # Set to True to save as GIF
    filename='kuramoto_evolution.gif'
)
animation_html  # Display the animation in the notebook


# In[32]:


def check_sphere_constraint(x_timeseries):
    """
    Check if oscillators remain on the unit sphere throughout the simulation.
    
    Parameters:
    - x_timeseries: ndarray of shape (num_times, n_osc, N)
        The time evolution of all oscillators
    
    Returns:
    - violation_stats: dict containing statistics about constraint violations
    """
    # Compute norms for all oscillators at all timesteps
    norms = np.linalg.norm(x_timeseries, axis=2)  # shape: (num_times, n_osc)
    
    # Check how far we deviate from unit norm
    deviations = np.abs(norms - 1.0)
    
    # Gather statistics
    max_deviation = np.max(deviations)
    mean_deviation = np.mean(deviations)
    worst_timestep = np.unravel_index(np.argmax(deviations), deviations.shape)[0]
    worst_oscillator = np.unravel_index(np.argmax(deviations), deviations.shape)[1]
    
    # Count significant violations (e.g., deviation > 0.01)
    significant_violations = np.sum(deviations > 0.01)
    total_points = deviations.size
    violation_percentage = (significant_violations / total_points) * 100
    
    # Create histogram data of deviations
    hist_counts, hist_edges = np.histogram(deviations, bins=50)
    
    violation_stats = {
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'worst_timestep': worst_timestep,
        'worst_oscillator': worst_oscillator,
        'violation_percentage': violation_percentage,
        'hist_counts': hist_counts,
        'hist_edges': hist_edges
    }
    
    print(f"Sphere Constraint Analysis:")
    print(f"Maximum deviation from unit norm: {max_deviation:.6f}")
    print(f"Mean deviation from unit norm: {mean_deviation:.6f}")
    print(f"Worst violation at timestep {worst_timestep}, oscillator {worst_oscillator}")
    print(f"Percentage of points with deviation > 0.01: {violation_percentage:.2f}%")
    
    return violation_stats

# Run the analysis on our simulation results
stats = check_sphere_constraint(x_timeseries)

# Visualize the distribution of deviations
plt.figure(figsize=(10, 6))
plt.bar(stats['hist_edges'][:-1], stats['hist_counts'], 
        width=np.diff(stats['hist_edges']), alpha=0.7)
plt.xlabel('Deviation from Unit Norm')
plt.ylabel('Count')
plt.title('Distribution of Deviations from Unit Sphere Constraint')
plt.yscale('log')  # Log scale to better see the distribution
plt.grid(True)
plt.show()

# Let's also look at how the maximum deviation evolves over time
max_deviations_over_time = np.max(np.abs(np.linalg.norm(x_timeseries, axis=2) - 1.0), axis=1)
plt.figure(figsize=(10, 6))
plt.plot(max_deviations_over_time)
plt.xlabel('Timestep')
plt.ylabel('Maximum Deviation from Unit Norm')
plt.title('Evolution of Maximum Sphere Constraint Violation')
plt.grid(True)
plt.show()

