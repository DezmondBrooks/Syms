import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ----------------------------
# Parameters
# ----------------------------
grid_size = 400
extent = 30
A0 = 1.0
decay_length = 5.0
k = 2 * np.pi
omega = 2 * np.pi
t_snapshot = 100
spacing = 2.27  # Approximate atomic spacing in iron
grid_count = 10  # 10x10 atoms
w_static = 1.0
w_wave = 1.0

# ----------------------------
# Generate grid
# ----------------------------
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
X, Y = np.meshgrid(x, y)
positions = np.stack((X, Y), axis=-1)  # shape: (H, W, 2)

# ----------------------------
# Generate lattice of effective atoms
# ----------------------------
def generate_lattice_of_atoms(grid_spacing, grid_count):
    atom_centers = []
    offset = grid_spacing * (grid_count // 2)
    for i in range(-grid_count // 2, grid_count // 2):
        for j in range(-grid_count // 2, grid_count // 2):
            atom_centers.append((i * grid_spacing, j * grid_spacing))
    return np.array(atom_centers)

atom_centers = generate_lattice_of_atoms(spacing, grid_count)

# ----------------------------
# Compute field from effective atom emitters
# ----------------------------
def compute_effective_atom_field(XY, source_positions, A0, k, decay_length, t, w_static, w_wave):
    H, W = XY.shape[:2]
    grid_flat = XY.reshape(-1, 2)  # shape: (H*W, 2)
    source_positions = source_positions[:, np.newaxis, :]  # shape: (N, 1, 2)
    r_vec = grid_flat - source_positions  # shape: (N, H*W, 2)
    r = np.linalg.norm(r_vec, axis=-1) + 1e-6  # avoid div/0
    decay = A0 * np.exp(-r / decay_length)
    combined = decay * (w_static + w_wave * np.cos(k * r - omega * t))  # shape: (N, H*W)
    field = np.sum(combined, axis=0).reshape(H, W)
    return field

# ----------------------------
# Compute and Plot
# ----------------------------
field = compute_effective_atom_field(
    positions, atom_centers, A0, k, decay_length, t_snapshot, w_static, w_wave
)

vmin, vmax = np.min(field), np.max(field)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, field, shading='auto', cmap='seismic', norm=norm)
plt.colorbar(label='Total Field Amplitude')
plt.title('Ripple Tank Field from Effective Atom Emitters (t = 100)\nEach Atom = Static + Wave Source')
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
