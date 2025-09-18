import fdtd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin

fdtd.set_backend("numpy")

# ----------------------------
# Grid
# ----------------------------
dx = 2e-3  # 2 mm
Nx, Ny, Nz = 20, 20, 100
grid = fdtd.Grid(shape=(Nx, Ny, Nz), grid_spacing=dx)

# ----------------------------
# Source & Detector positions
# ----------------------------
source_z = 0.02      # 2 cm
detector_z = Nz*dx*0.5    # 0.9 times z length
source_cell = int(source_z / dx)
detector_cell = int(detector_z / dx)

# ----------------------------
# FDTD parameters
# ----------------------------
c0 = 3e8
dt = dx / (2*c0)      # CFL
tau_fwhm = 5e-9       # 5 ns
f = 1e9               # 1 GHz
E0 = 1e4              # V/m

# 传播时间, 总模拟时间步数
t_prop = detector_z - source_z
n_steps = int((t_prop / c0 + 2*tau_fwhm) / dt)

# Hanning pulse对应的周期步数
cycle_steps = int(10 * tau_fwhm / dt)

# ----------------------------
# Source
# ----------------------------
src = fdtd.PointSource(
    period=int(1/f/dt),  # 1 GHz
    amplitude=E0,
    pulse=True,
    cycle=cycle_steps,
    name="microwave_pulse"
)
grid[Nx//2, Ny//2, source_cell] = src

# ----------------------------
# Detector
# ----------------------------
grid[:, :, detector_cell] = fdtd.LineDetector(name="probe")

# ----------------------------
# PML
# ----------------------------
pml = 10
grid[0:pml, :, :] = fdtd.PML(name="pml_xlow")
grid[-pml:, :, :] = fdtd.PML(name="pml_xhigh")
grid[:, 0:pml, :] = fdtd.PML(name="pml_ylow")
grid[:, -pml:, :] = fdtd.PML(name="pml_yhigh")
grid[:, :, 0:pml] = fdtd.PML(name="pml_zlow")
grid[:, :, -pml:] = fdtd.PML(name="pml_zhigh")

# ----------------------------
# Run
# ----------------------------
grid.run(total_time=n_steps)

# ----------------------------
# Extract signal
# ----------------------------
detector = grid.detectors[0]
E_data = np.array(detector.E)
# Ez = E_data[:, E_data.shape[1]//2, 2]
Ez = np.max(E_data[:, :, 2], axis=1)  # 或 np.mean()
time = np.arange(len(Ez)) * dt

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,4))
plt.plot(time*1e9, Ez)
plt.xlabel("Time (ns)")
plt.ylabel("Ez (V/m)")
plt.title("Probe signal at z={:.2f} m".format(detector_z))
plt.grid()
plt.show()
