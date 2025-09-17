"""
2D axisymmetric (r-z) FDTD (TM^phi): Ez, Er, Hphi
Axisymmetric baseline: wave propagates along +z for 1 m, plasma region at z=0.5 m but density = 0 (baseline).

Notes:
- Units: SI
- This is a demonstration/educational implementation. For production quality use a validated cylindrical-FDTD code/PML.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Physical constants
eps0 = 8.8541878128e-12
mu0 = 4*np.pi*1e-7
c0 = 1/np.sqrt(eps0*mu0)

# Simulation domain (meters)
Rmax = 0.02      # radial extent 2 cm (choose > plasma radius). You can reduce.
Lz = 1.0         # propagation length 1 m as requested

# Grid resolution
dr = 2e-4        # 0.2 mm radial step (tune for speed/accuracy)
dz = 2e-4        # 0.2 mm axial step
Nr = int(Rmax/dr)
Nz = int(Lz/dz)

print(f"Grid: Nr={Nr}, Nz={Nz}, dr={dr}, dz={dz}")

# CFL time step (2D cylindrical uses same form as 2D Cartesian for dt)
dt = 0.99 / (c0 * np.sqrt((1/dr**2) + (1/dz**2)))
print(f"dt = {dt:.3e} s")

# Time: choose total time to see pulse reach probe at z=0.9 m
t_total = 5e-8   # default 50 ns; increase if you need longer observation
n_steps = int(np.ceil(t_total / dt))
print(f"n_steps = {n_steps}")

# Field arrays (using double precision)
# Ez defined at r centers (i=0..Nr-1) and z nodes (j=0..Nz-1)
Ez = np.zeros((Nr, Nz))
# Er defined at radial edges r = i*dr (i=0..Nr), but we store 0..Nr-1 and enforce Er[0,:]=0
Er = np.zeros((Nr+1, Nz))
# Hphi defined at r-centers and half z-step (we store same shape as Ez for simplicity)
Hphi = np.zeros((Nr, Nz))

# r coordinates for Ez indices (center of cell): r_i = (i+0.5)*dr
r = (np.arange(Nr) + 0.5) * dr
# r coordinates for Er indices (edge): r_edge = i * dr, i=0..Nr
r_edge = np.arange(Nr+1) * dr

# Source parameters
f0 = 1e9
omega0 = 2*np.pi*f0
E0 = 1e4           # 10 kV/m
pulse_width = 5e-9 # 5 ns envelope for demo
t0 = 4 * pulse_width

# Source location: inject a plane wave (axisymmetric) at z_src (a z-index)
z_src = int(0.05 / dz)   # z = 0.05 m
# For axisymmetry, inject across all r (a flat wavefront) or only near axis? we inject across r
# Probe location
z_probe = int(0.9 / dz)
r_probe_idx = 0   # axis (r=0) probe

# For recording
src_record = np.zeros(n_steps)
probe_record = np.zeros(n_steps)

# Simple Mur boundary helper for Ez at z boundaries (first order)
def mur_z(E):
    # E shape (Nr, Nz)
    E[:,0] = E[:,1]
    E[:,-1] = E[:,-2]
    return E

# Time stepping
for n in range(n_steps):
    t = n * dt

    # 1) Update Hphi at half-step: Hphi^{n+1/2} = Hphi^{n-1/2} + (dt/mu0)*(dEr/dz - dEz/dr)
    # dEr/dz: (Er[:, j+1] - Er[:, j]) / dz  but Er has Nz points at same z nodes.
    # Use central differences where possible.
    # For Ez radial derivative: dEz/dr at r-centers approximated by (Ez[i,j] - Ez[i-1,j])/dr with special at i=0
    # We'll compute arrays for dEr_dz and dEz_dr
    dEr_dz = np.zeros_like(Hphi)
    dEz_dr = np.zeros_like(Hphi)

    # dEr/dz for i=0..Nr (use Er[ i , j+1] - Er[i, j] ), but Hphi is defined for i=0..Nr-1 and j=0..Nz-1
    # so take Er[i, j+1] - Er[i, j] -> produce for j=0..Nz-2; for last z use backward diff
    dEr_dz[:, :-1] = (Er[:Nr, 1:] - Er[:Nr, :-1]) / dz
    dEr_dz[:, -1] = (Er[:Nr, -1] - Er[:Nr, -2]) / dz

    # dEz/dr: for i=0 use forward diff (symmetry -> dEz/dr|_{r->0} = 0)
    # for i>=1 central/backward:
    # approximate dEz/dr at center r_i by (Ez[i,j] - Ez[i-1,j]) / dr
    dEz_dr[0, :] = 0.0
    if Nr > 1:
        dEz_dr[1:, :] = (Ez[1:, :] - Ez[:-1, :]) / dr

    Hphi += (dt/mu0) * (dEr_dz - dEz_dr)

    # 2) Update Er: Er^{n+1} = Er^{n} - (dt/eps0) * dHphi/dz
    # dHphi/dz approximated at Er locations (z nodes): (Hphi[:, j] - Hphi[:, j-1])/dz with j=0..Nz-1
    dH_dz = np.zeros_like(Er[:Nr, :])
    # for j=0 use forward difference
    dH_dz[:,0] = (Hphi[:,0] - 0.0) / dz   # assume Hphi[:, -1] beyond is zero initially; okay for short times
    if Nz > 1:
        dH_dz[:,1:] = (Hphi[:,1:] - Hphi[:,:-1]) / dz

    Er[:Nr, :] -= (dt/eps0) * dH_dz
    # enforce symmetry at axis: Er[0,:] = 0
    Er[0, :] = 0.0

    # 3) Update Ez: Ez^{n+1} = Ez^{n} + (dt/eps0) * (1/r) * d( r * Hphi )/dr
    # Compute r*Hphi at radial cell edges: need values at r_edges k=0..Nr (we'll construct)
    rH_edge = np.zeros((Nr+1, Nz))
    # rH_edge[1..Nr-1] ~ r_edge * Hphi at neighboring center; approximate:
    # place Hphi at center r = (i+0.5)dr; for edge k, use average of adjacent Hphi
    # rH_edge[k] = r_edge[k] * avg(Hphi[k-1], Hphi[k]) for k=1..Nr-1
    # For k=0 (axis), use rH_edge[0] = 0
    rH_edge[0, :] = 0.0
    if Nr > 1:
        # for k = 1..Nr-1
        rH_edge[1:Nr, :] = (r_edge[1:Nr, None]) * 0.5*(Hphi[:Nr-1, :] + Hphi[1:Nr, :])
    # for last edge k = Nr, use mirrored value of last Hphi
    rH_edge[Nr, :] = r_edge[Nr] * Hphi[-1, :]

    # radial derivative: (rH_edge[k+1] - rH_edge[k]) / (dr) evaluated at Ez cell i -> use k=i
    dr_term = (rH_edge[1:Nr+1, :] - rH_edge[:Nr, :]) / dr   # shape (Nr, Nz)
    # divide by r (Ez cell r = (i+0.5)*dr)
    Ez += (dt/eps0) * (dr_term / r[:, None])

    # Apply simple Mur boundary at z ends for Ez
    Ez = mur_z(Ez)

    # Source injection: soft source on Ez at z_src across r (plane wave front)
    envelope = np.exp(-((t - t0)**2) / (2 * pulse_width**2))
    src_val = E0 * envelope * np.sin(omega0 * t)
    # inject across all r (axisymmetric plane front)
    Ez[:, z_src] += src_val

    # record
    src_record[n] = src_val
    probe_record[n] = Ez[r_probe_idx, z_probe]

    # optional: print early max field to debug early-time nonlocal injection
    if n < 6:
        print(f"n={n}, max|Ez|={np.max(np.abs(Ez)):.3e}")

print("Simulation finished.")

# Plot source and probe time series
time_ns = np.arange(n_steps) * dt * 1e9

plt.figure(figsize=(9,3))
plt.plot(time_ns, src_record, label='source (Ez injection)')
plt.plot(time_ns, probe_record, label=f'probe at r=0,z={z_probe*dz:.3f} m')
plt.xlabel("Time (ns)")
plt.ylabel("Ez (V/m)")
plt.legend()
plt.grid(True)
plt.title("Source vs Probe (axisymmetric baseline)")
plt.show()

# Show a snapshot of Ez (r-z) at final time
plt.figure(figsize=(6,4))
extent = [0, Lz, 0, Rmax]  # imshow extent: x->z, y->r
plt.imshow(Ez.T, origin='lower', extent=extent, aspect='auto', cmap='RdBu')
plt.colorbar(label='Ez (V/m)')
plt.xlabel('z (m)')
plt.ylabel('r (m)')
plt.title('Ez snapshot (r-z plane) at final time')
plt.show()
