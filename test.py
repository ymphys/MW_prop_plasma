"""
2D FDTD (TEz) simulation of a plane microwave pulse (1 GHz) interacting with
a circular plasma region (radius 1 cm) centered at x=0.5 m.

Notes:
- The plasma is modeled with the Drude equation: dJ/dt + nu * J = eps0 * wp^2 * E
- wp (plasma frequency) computed from local electron density n_e(r).
- For performance the pulse width is set shorter by default. Change pulse_width to 1e-6
  if you understand the computational cost (will require many more timesteps).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Physical constants
eps0 = 8.8541878128e-12
mu0 = 4*np.pi*1e-7
c0 = 1/np.sqrt(eps0*mu0)
qe = 1.602176634e-19
me = 9.10938356e-31
# print(c0)

# Simulation domain (meters)
Lx = 1.0        # total length in x (user specified)
Ly = 0.2        # choose a small y-extent to include 1 cm plasma radius comfortably
# grid resolution
dx = 2e-3       # 2 mm spatial step (tune for accuracy vs speed)
dy = dx
Nx = int(Lx/dx)
Ny = int(Ly/dy)

# Time stepping
# CFL condition for 2D: dt <= 1/(c * sqrt( (1/dx^2) + (1/dy^2) ))
dt = 0.99 / (c0 * np.sqrt((1/dx**2)+(1/dy**2)))
print(f"Grid: {Nx} x {Ny}, dt = {dt:.3e} s")

# Wave / source parameters
f0 = 1e9
omega0 = 2*np.pi*f0
# pulse parameters
pulse_center = 50*dt   # in timesteps (we will use gaussian in time)
# For demonstration we use a shorter envelope than 1e-6 to keep runtime small.
pulse_width = 50e-9    # 50 ns by default; change to 1e-6 if you accept heavy computation
# amplitude
E0 = 1e4   # 10 kV/m
# print(pulse_center)

# Convert gaussian width to timesteps (for source generation)
t0 = 6 * pulse_width
t_total = t0 + 6 * pulse_width   # 覆盖到 t0 + 6σ
# t_total = 2000 * dt   # total sim time (s) -- tune as needed
n_steps = int(np.ceil(t_total / dt))
print(f"Pulse t0 = {t0:.3e} s, total sim time = {t_total:.3e} s, n_steps = {n_steps}")

# Prepare fields: TEz (Ez, Hx, Hy)
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

# Current density Jz for Drude plasma model (per-cell)
Jz = np.zeros((Nx, Ny))

# Electron density map (circular gaussian)
# Plasma center at x = 0.5 m, y = Ly/2
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
# plasma geometry
r_center = 0.5
y_center = Ly/2.0
r_grid = np.sqrt((X - r_center)**2 + (Y - y_center)**2)
r_edge = 0.01  # 1 cm radius

n0 = 1e16      # center density
n_edge = 1e12  # density at r = r_edge
# Solve for sigma of gaussian radial profile: n(r) = n0 * exp(-r^2/(2 sigma^2))
sigma = np.sqrt(- (r_edge**2) / (2*np.log(n_edge / n0)))
print(f"Computed gaussian sigma = {sigma:.3e} m")

# free space
n_e = np.zeros_like(X)

# Uniform distribution
# n_e = np.full_like(X, 1e16)


# Gaussian profile with cutoff
# n_e = n0 * np.exp(-r_grid**2 / (2*sigma**2))
# Outside radius, we can smoothly go to background (but gaussian already does)
# Optionally set floor:
# n_e[n_e < n_edge] = n_edge

# Plasma parameters per cell: plasma frequency squared
wp2 = (n_e * qe**2) / (eps0 * me)   # omega_p^2
# Collision frequency (nu) - user did not specify; choose modest collision freq
nu = 1e7  # s^-1, can be tuned

# Precompute update coefficients for J ODE discretization (explicit Euler)
# dJ/dt = eps0*wp^2 * E - nu * J
# J^{n+1} = J^n + dt*(eps0*wp^2*E - nu*J^n)
# We will use a semi-implicit form for a bit more stability:
# J^{n+1} = (1 - nu*dt) / (1 + nu*dt) * J^n + (eps0*wp^2*dt) / (1 + nu*dt) * E^{n+1/ n avg}
# For simplicity use explicit Euler here (sufficient for small dt)
alpha_j = eps0 * wp2
# Source position: left side injecting plane wave (line source across y)
src_x = int(0.05 / dx)   # put source at x=5 cm
src_y_indices = np.arange(Ny)  # whole y (plane wave)

# For recording
probe_pos = (int(0.9/dx), int(Ny/2))  # probe near the right side
Ez_probe = np.zeros(n_steps)
# 用一个数组记录源（用于诊断）
src_record = np.zeros(n_steps)

# Simple Mur absorbing boundary 1st order coefficients for Ez (left/right/top/bottom)
# Implementation: apply Mur after Ez updated (works reasonably for demonstration)
def mur_update(E):
    # left/right (x direction)
    # For simplicity we use zero-gradient first-order absorbing (not perfect)
    E[0, :] = E[1, :]
    E[-1, :] = E[-2, :]
    # top/bottom (y direction)
    E[:, 0] = E[:, 1]
    E[:, -1] = E[:, -2]
    return E

# Coefficients for curl updates
# Update Hx, Hy from Ez:
# Hx^{n+1/2} = Hx^{n-1/2} - (dt/mu0) * dEz/dy
# Hy^{n+1/2} = Hy^{n-1/2} + (dt/mu0) * dEz/dx

# Update Ez from H:
# Ez^{n+1} = Ez^{n} + (dt/eps0) * (dHy/dx - dHx/dy) - (dt/eps0) * Jz

# Precompute constants
coef_H = dt / mu0
coef_E = dt / eps0

# Prepare plotting / animation
fig, ax = plt.subplots(figsize=(8,4))
im = ax.imshow(Ez.T, origin='lower', extent=[0,Lx,0,Ly], cmap='RdBu', vmin=-E0, vmax=E0)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Ez (V/m)')
plt.colorbar(im, ax=ax)

# Time-stepping main loop
frame_interval = max(1, n_steps // 100)  # for animation updates

for n in range(n_steps):
    # Update H fields (centered half-step)
    # Hx[i,j] -= coef_H * (Ez[i, j+1] - Ez[i, j]) / dy
    Hx[:, :-1] -= coef_H * (Ez[:, 1:] - Ez[:, :-1]) / dy
    # Hy[i,j] += coef_H * (Ez[i+1, j] - Ez[i, j]) / dx
    Hy[:-1, :] += coef_H * (Ez[1:, :] - Ez[:-1, :]) / dx

    # Source: we inject into Ez later (hard source or soft source),
    # here use a soft source by adding to Ez at src plane
    # Temporal Gaussian-modulated sinusoid:
    t = n * dt
    # Gaussian envelope centered at t0 with width pulse_width
    envelope = np.exp(-((t - t0)**2) / (2*(pulse_width**2)))
    src_val = E0 * envelope * np.sin(omega0 * t)
    src_record[n] = src_val
    # Inject across the whole src_x column (plane wave)
    Ez[src_x, :] += src_val

    # Update J (explicit Euler)
    # dJ/dt = eps0*wp2 * E - nu * J
    Jz += dt * (alpha_j * Ez - nu * Jz)

    # Update Ez from H and J
    # Ez[1:-1,1:-1] += coef_E * ( (Hy[1:,1:-1]-Hy[:-1,1:-1])/dx - (Hx[1:-1,1:]-Hx[1:-1,:-1])/dy ) - (dt/eps0)*Jz
    curl_H = np.zeros_like(Ez)
    curl_H[1:-1,1:-1] = (Hy[1:-1,1:-1] - Hy[:-2,1:-1]) / dx - (Hx[1:-1,1:-1] - Hx[1:-1,:-2]) / dy
    Ez += coef_E * curl_H - (dt/eps0) * Jz

    # Apply simple absorbing boundaries
    Ez = mur_update(Ez)

    # record probe
    Ez_probe[n] = Ez[probe_pos]

    # visualization update
    if (n % frame_interval) == 0:
        im.set_data(Ez.T)
        ax.set_title(f"Ez (V/m), t = {t*1e9:.2f} ns")
        plt.pause(0.001)

print("Simulation finished.")

# 时间轴（秒）
time = np.arange(n_steps) * dt

# normalize (去直流并归一化) 以便交叉相关更稳健
src = src_record - np.mean(src_record)
probe = Ez_probe - np.mean(Ez_probe)

# 交叉相关（使用 FFT 更快）
corr = np.fft.ifft(np.fft.fft(probe, n=2*n_steps) * np.conj(np.fft.fft(src, n=2*n_steps)))
corr = np.real(corr)
lags = np.arange(-n_steps, n_steps) * dt  # seconds
# take middle portion corresponding to valid lags
corr_valid = np.concatenate((corr[-n_steps+1:], corr[:n_steps]))
# find lag at max correlation
lag_index = np.argmax(corr_valid)
measured_lag = lags[lag_index]
print(f"Measured lag = {measured_lag:.3e} s = {measured_lag*1e9:.3f} ns")

# 画图：源、探针、探针（按测得延迟反移）
plt.figure(figsize=(9,5))
plt.plot(time*1e9, src_record, label='source (injected)', alpha=0.8)
plt.plot(time*1e9, Ez_probe, label='probe (raw)', alpha=0.8)
# shift probe back by measured lag to align with source
shift_samples = int(round(measured_lag / dt))
if abs(shift_samples) < len(time):
    probe_shifted = np.roll(Ez_probe, -shift_samples)
    plt.plot(time*1e9, probe_shifted, '--', label=f'probe (shifted by {measured_lag*1e9:.2f} ns)')
else:
    print("shift_samples out of range; skipping rolled plot")

plt.xlabel("Time (ns)")
plt.ylabel("Ez (V/m)")
plt.legend()
plt.grid(True)
plt.title("Source vs Probe (measured lag shown)")
plt.savefig("source_vs_probe.jpeg")

# Visualize electron density map and plasma wp map
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(n_e.T, origin='lower', extent=[0,Lx,0,Ly], cmap='viridis', norm=None)
plt.colorbar(label='n_e (m^-3)')
plt.title('Electron density n_e')
plt.subplot(1,2,2)
plt.imshow(np.sqrt(wp2).T/(2*np.pi), origin='lower', extent=[0,Lx,0,Ly], cmap='inferno')
plt.colorbar(label='f_p = omega_p/(2π) (Hz)')
plt.title('Local plasma frequency (Hz)')
plt.show()
plt.savefig("plasma.jpeg")

# 计算源 & 探针在空间中的位置（米）
src_x_index = src_x                      # 从你的仿真变量取出
probe_x_index = probe_pos[0]
print("src_x index:", src_x_index, "probe_x index:", probe_x_index)
print("src_x pos (m):", src_x_index * dx, "probe_x pos (m):", probe_x_index * dx)
dist_m = (probe_x_index - src_x_index) * dx
print("distance (m) probe <- source:", dist_m)

# 预期传播延迟
t_delay_expected = dist_m / c0
print(f"expected time delay = {t_delay_expected:.3e} s = {t_delay_expected*1e9:.3f} ns")

# 打印 src_record 和 Ez_probe 的峰值大小对比
print("max(src_record) = ", np.max(np.abs(src_record)))
print("max(Ez_probe) = ", np.max(np.abs(Ez_probe)))

from scipy.signal import hilbert

# time axis
n_steps = len(src_record)
time = np.arange(n_steps) * dt
time_ns = time * 1e9

# spatial positions (meters)
src_x_pos = src_x * dx
probe_x_pos = probe_pos[0] * dx
dist = probe_x_pos - src_x_pos
theory_delay = dist / c0

print(f"source x = {src_x_pos:.6f} m, probe x = {probe_x_pos:.6f} m, distance = {dist:.6f} m")
print(f"theoretical propagation delay d/c = {theory_delay*1e9:.6f} ns")

# -----------------------------
# Compute analytic envelope (Hilbert)
# -----------------------------
analytic_src = hilbert(src_record)
env_src = np.abs(analytic_src)

analytic_probe = hilbert(Ez_probe)
env_probe = np.abs(analytic_probe)

# normalize envelopes for plotting convenience
env_src_n = env_src / np.max(env_src)
env_probe_n = env_probe / np.max(env_probe)

# -----------------------------
# Method A: envelope peak times
# -----------------------------
idx_src_peak = np.argmax(env_src)
idx_probe_peak = np.argmax(env_probe)
t_src_peak = idx_src_peak * dt
t_probe_peak = idx_probe_peak * dt
delay_peak = t_probe_peak - t_src_peak

print("\nMethod A - Envelope peak:")
print(f"  src peak index {idx_src_peak}, time {t_src_peak*1e9:.6f} ns")
print(f"  probe peak index {idx_probe_peak}, time {t_probe_peak*1e9:.6f} ns")
print(f"  measured group delay (peak) = {delay_peak*1e9:.6f} ns")

# -----------------------------
# Method B: cross-correlation of envelopes
# -----------------------------
# remove mean to avoid DC bias
src_env_zero = env_src - np.mean(env_src)
probe_env_zero = env_probe - np.mean(env_probe)

corr = np.correlate(probe_env_zero, src_env_zero, mode='full')
lags = np.arange(-len(src_env_zero)+1, len(src_env_zero)) * dt
lag_idx = np.argmax(corr)
measured_env_lag = lags[lag_idx]

print("\nMethod B - Envelope cross-correlation:")
print(f"  measured group delay (corr) = {measured_env_lag*1e9:.6f} ns")

# -----------------------------
# Method C: threshold crossing (first time envelope exceeds frac*peak)
# -----------------------------
frac = 0.1  # 阈值，采用峰值的 10% 作为到达判定。可改为 0.05 或 0.2 做稳健检验
th_src = frac * np.max(env_src)
th_probe = frac * np.max(env_probe)

# first index where envelope >= threshold
def first_crossing_index(env, thresh):
    idxs = np.where(env >= thresh)[0]
    return int(idxs[0]) if idxs.size>0 else None

i_src_th = first_crossing_index(env_src, th_src)
i_probe_th = first_crossing_index(env_probe, th_probe)

if i_src_th is None or i_probe_th is None:
    print("\nMethod C - Threshold crossing: 未找到阈值穿越 (检查阈值或信号)")
    delay_thresh = None
else:
    t_src_th = i_src_th * dt
    t_probe_th = i_probe_th * dt
    delay_thresh = t_probe_th - t_src_th
    print("\nMethod C - Threshold crossing:")
    print(f"  threshold fraction = {frac:.3f}")
    print(f"  src first crossing idx {i_src_th}, time {t_src_th*1e9:.6f} ns")
    print(f"  probe first crossing idx {i_probe_th}, time {t_probe_th*1e9:.6f} ns")
    print(f"  measured group delay (threshold) = {delay_thresh*1e9:.6f} ns")

# -----------------------------
# Summarize numeric comparison
# -----------------------------
print("\nSummary (all in ns):")
print(f"  theoretical delay = {theory_delay*1e9:.6f} ns")
print(f"  delay (peak)       = {delay_peak*1e9:.6f} ns")
print(f"  delay (corr env)   = {measured_env_lag*1e9:.6f} ns")
if delay_thresh is not None:
    print(f"  delay (threshold)  = {delay_thresh*1e9:.6f} ns")

# -----------------------------
# Plot envelopes and indicate measured delays
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(time_ns, env_src_n, label='env source (normalized)', lw=2)
plt.plot(time_ns, env_probe_n, label='env probe (normalized)', lw=2)
# mark peaks
plt.axvline(t_src_peak*1e9, color='C0', ls='--', label=f'src peak {t_src_peak*1e9:.2f} ns')
plt.axvline(t_probe_peak*1e9, color='C1', ls='--', label=f'probe peak {t_probe_peak*1e9:.2f} ns')
# mark cross-corr alignment (place vertical at src_peak + measured_env_lag)
plt.axvline((t_src_peak + measured_env_lag)*1e9, color='k', ls=':', label=f'corr align {measured_env_lag*1e9:.2f} ns')
# mark threshold crossing times if available
if i_src_th is not None:
    plt.axvline(i_src_th*dt*1e9, color='C0', ls=':', label=f'src threshold {i_src_th*dt*1e9:.2f} ns')
if i_probe_th is not None:
    plt.axvline(i_probe_th*dt*1e9, color='C1', ls=':', label=f'probe threshold {i_probe_th*dt*1e9:.2f} ns')

plt.xlim(0, min(time_ns[-1], (t_probe_peak+5e-8)*1e9))  # 视图只看到 probe 峰附近
plt.xlabel('Time (ns)')
plt.ylabel('Envelope (normalized)')
plt.title('Envelopes: source vs probe and measured delays')
plt.legend()
plt.grid(True)
plt.show()
