import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Grid and simulation setup
grid_size = 400
extent = 30
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
X, Y = np.meshgrid(x, y)
positions = np.stack((X, Y), axis=-1)

# Constants
A0 = 10
wavelength = 1.0
k = 2 * np.pi / wavelength
omega = 2 * np.pi
t_snapshot = 100
spacing = 2.27
grid_count = 10

# Generate lattice of atoms
def generate_lattice_of_atoms(grid_spacing, grid_count):
    atom_centers = []
    for i in range(-grid_count // 2, grid_count // 2):
        for j in range(-grid_count // 2, grid_count // 2):
            atom_centers.append((i * grid_spacing, j * grid_spacing))
    return np.array(atom_centers)

atom_centers = generate_lattice_of_atoms(spacing, grid_count)

# Compute EM-like intensity field
def compute_em_intensity_field(XY, source_positions, A0, k, t):
    H, W = XY.shape[:2]
    grid_flat = XY.reshape(-1, 2)
    source_positions = source_positions[:, np.newaxis, :]
    r_vec = grid_flat - source_positions
    r = np.linalg.norm(r_vec, axis=-1) + 1e-6
    amplitude = A0 / r * np.cos(k * r - omega * t)
    total_amplitude = np.sum(amplitude, axis=0).reshape(H, W)
    intensity = total_amplitude ** 2
    return intensity

# Compute intensity
intensity_field = compute_em_intensity_field(positions, atom_centers, A0, k, t_snapshot)

# Print range of intensity values
print("Intensity min/max:", np.min(intensity_field), np.max(intensity_field))

# Use logarithmic color scale if range is tight
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, intensity_field, shading='auto', cmap='inferno',
               norm=LogNorm(vmin=1e-6, vmax=np.max(intensity_field)))
plt.colorbar(label='Log-scaled EM Intensity (∝ Amplitude²)')
plt.title('Log EM Intensity Field from Effective Iron Atom Emitters')
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
