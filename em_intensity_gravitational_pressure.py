import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Simulation Grid Setup ---
grid_size = 400
extent = 30
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
X, Y = np.meshgrid(x, y)
positions = np.stack((X, Y), axis=-1)

# --- Constants ---
A0 = 1.0
wavelength = 1.0
k = 2 * np.pi / wavelength
omega = 2 * np.pi
t_snapshot = 100
spacing = 2.27
grid_count = 10

# --- Physics Constants for Comparison ---
G = 6.67430e-11       # m³/kg/s²
m_fe = 9.27e-26       # kg (mass of one iron atom)
rho_iron = 7870       # kg/m³
c_sound = 5000        # m/s (sound speed in iron)

# --- Generate Atom Lattice ---
def generate_lattice_of_atoms(grid_spacing, grid_count):
    atom_centers = []
    for i in range(-grid_count // 2, grid_count // 2):
        for j in range(-grid_count // 2, grid_count // 2):
            atom_centers.append((i * grid_spacing, j * grid_spacing))
    return np.array(atom_centers)

atom_centers = generate_lattice_of_atoms(spacing, grid_count)

# --- Compute EM-Like Intensity Field ---
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

intensity_field = compute_em_intensity_field(positions, atom_centers, A0, k, t_snapshot)

# --- Pick a Reference Point ---
r_probe = 10.0  # radius from center
x_probe = int((r_probe + extent) / (2 * extent) * grid_size)
y_probe = grid_size // 2
intensity_at_probe = intensity_field[y_probe, x_probe]

# --- Gravitational Field and Pressure at Same Point ---
total_atoms = grid_count ** 2
total_mass = total_atoms * m_fe
g_gravity = G * total_mass / (r_probe ** 2)
u_g = g_gravity ** 2 / (8 * np.pi * G)
P_gravity = u_g / (2 * rho_iron * c_sound ** 2)

# --- Output Comparison ---
print(f"EM Wave Intensity at r = {r_probe} units: {intensity_at_probe:.3e}")
print(f"Gravitational field strength at r = {r_probe} m: {g_gravity:.3e} m/s²")
print(f"Gravitational energy density: {u_g:.3e} J/m³")
print(f"Gravitational pressure equivalent: {P_gravity:.3e} Pa")

# --- Plot Intensity Heatmap ---
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, intensity_field, shading='auto', cmap='inferno', norm=LogNorm(vmin=1e-6, vmax=np.max(intensity_field)))
plt.colorbar(label='Log EM Wave Intensity (∝ Amplitude²)')
plt.scatter(x[x_probe], y[y_probe], color='cyan', label=f"r = {r_probe} probe")
plt.legend()
plt.title('EM-Like Intensity Field from Effective Iron Atom Emitters')
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
