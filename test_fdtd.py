import fdtd
import numpy as np
import matplotlib.pyplot as plt

# Create a grid
grid = fdtd.Grid(
    shape=(100, 100, 1),  # 2D grid (100x100)
    grid_spacing=1e-3,    # 1 mm grid spacing
    permittivity=1.0,     # vacuum
    permeability=1.0,     # vacuum
)

# Add a line source
grid[50, 10, 0] = fdtd.LineSource(
    period=20,
    name="source"
)

# Add a detector
grid[50, 90, 0] = fdtd.LineDetector(name="detector")

# Add boundary conditions (official fdtd style)
# X boundaries: 10-cell PML
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
# Y boundaries: 10-cell PML
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")
# Z boundary: periodic (for 2D grid, z is length 1)
grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

# Run the simulation
for _ in range(200):
    grid.step()

# Plot the detector signal
plt.plot(grid.detector_signals("detector"))
plt.xlabel("Time step")
plt.ylabel("Signal")
plt.title("Detector Signal vs Time Step")
plt.show()
