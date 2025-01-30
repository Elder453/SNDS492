#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from PIL import Image

from kuramoto import GeneralizedKuramoto


# In[27]:


# ----------------------------------------------------------------------
# Step 1: Define Problem Size and Load Fish Mask
# ----------------------------------------------------------------------
H, W = 64, 64            # height and width of the oscillator grid
n_osc = H * W            # total number of oscillators
N = 2                    # dimension of each oscillator's state vector

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
c[fish_mask_flat > 0.5] = 1.0

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

# Rotational coupling matrix (N=2)
base_matrix = np.eye(2, dtype=np.float32) # shape: (2, 2)

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
base_omega = np.array([[0, omega_val], [-omega_val, 0]], dtype=np.float32)

# Initialize Omega with spatially varying natural frequencies
Omega = np.zeros((n_osc, N, N), dtype=np.float32) # shape: (4096, 2, 2)
for i in range(n_osc):
    # Add randomness to natural frequencies (optional)
    Omega[i] = base_omega * (1 + 0.01 * np.random.randn())


# In[28]:


# ----------------------------------------------------------------------
# Step 5: Initialize the GeneralizedKuramoto Model
# ----------------------------------------------------------------------
gamma = 1    # step size for updates
T = 250         # total number of discrete steps

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
    """
    Compute the energy E(t) at each time step.

    Parameters:
    - x_series: ndarray of shape (num_times, n_osc, N)
    - J: ndarray of shape (n_osc, n_osc)
    - c: ndarray of shape (n_osc, N)

    Returns:
    - E: ndarray of shape (num_times,)
    """
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

E = compute_energy(x_timeseries, J, c)  # shape: (num_times,)


# In[30]:


# ----------------------------------------------------------------------
# Step 9: Two-row Visualization with Mask and Evolution
# ----------------------------------------------------------------------
def map_states_to_rgb(X, H, W):
    """
    Convert oscillator states to RGB colors by mapping vector components directly.

    First component -> Red channel
    Second component -> Green channel
    Fixed value -> Blue channel

    Parameters:

    - X: ndarray of shape (n_osc, 2), oscillator states on unit sphere
    - H, W: integers defining the grid dimensions
    
    Returns:
    - rgb: ndarray of shape (H, W, 3) containing RGB values in [0,1] range
    """

    # Reshape the states to match the grid structure
    states = X.reshape((H, W, 2))
    
    # Scale from [-1,1] to [0,1] range since states are on unit sphere
    states_norm = (states + 1) / 2
    
    # Construct RGB array with fixed blue value
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = states_norm[:, :, 0]  # X component -> Red
    rgb[:, :, 1] = states_norm[:, :, 1]  # Y component -> Green
    rgb[:, :, 2] = 0.5                   # Fixed blue value for consistent appearance

    return rgb

def plot_energy_and_evolution(E, x_timeseries, fish_mask, H, W, gamma, T):
    """
    Create a two-row visualization with:
    Top row: Energy plot spanning columns 2-6
    Bottom row: Mask in column 1, followed by 5 oscillator state snapshots
    
    Parameters:
    - E: ndarray, energy values over time
    - x_timeseries: ndarray, shape (num_times, n_osc, 2) containing oscillator states
    - fish_mask: ndarray, shape (H, W) containing the binary mask
    - H, W: integers, height and width of the grid
    - gamma: float, time step size
    - T: int, total number of time steps
    """
    # Create figure with 2x6 grid layout
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])
    
    # First row: Empty first column, Energy plot spans columns 2-6
    ax_energy = plt.subplot(gs[0, 1:])  # Note: starts at column 1 (second column)
    time_points = np.arange(len(E)) * gamma
    ax_energy.plot(time_points, E, 'b-', label='Energy E(t)')
    ax_energy.set_xlabel('Time')
    ax_energy.set_ylabel('Energy E(t)')
    ax_energy.grid(True)
    ax_energy.legend()
    
    # Second row: First show the mask
    ax_mask = plt.subplot(gs[1, 0])  # First column of second row
    ax_mask.imshow(fish_mask, cmap='gray')
    ax_mask.set_title('Mask')
    ax_mask.axis('off')
    
    # Calculate 5 evenly spaced snapshot times for the remaining columns
    snapshot_steps = np.linspace(0, T-1, 5, dtype=int)
    
    # Add vertical lines to energy plot for snapshot times
    for step in snapshot_steps:
        time = step * gamma
        ax_energy.axvline(x=time, color='r', linestyle='--', alpha=0.3)
    
    # Plot oscillator states in columns 2-6 of second row
    for idx, step in enumerate(snapshot_steps):
        ax_state = plt.subplot(gs[1, idx+1])  # +1 because mask takes first column
        X_snapshot = x_timeseries[step]
        rgb = map_states_to_rgb(X_snapshot, H, W)
        ax_state.imshow(rgb, interpolation='nearest')
        ax_state.set_title(f't = {step * gamma:.2f}')
        ax_state.axis('off')
    
    plt.suptitle('Kuramoto Oscillator Dynamics', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show();

# Create the visualization
plot_energy_and_evolution(E, x_timeseries, fish_mask, H, W, gamma, T);


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

