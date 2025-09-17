import numpy as np
import matplotlib.pyplot as plt

# --- 基本参数 ---
c0 = 3e8            # 光速
eps0 = 8.854e-12
mu0 = 4*np.pi*1e-7

# 空间网格
Lx, Ly, Lz = 0.1, 0.1, 1.0     # 尺寸 (x,y,z)，单位 m
dx = dy = dz = 2e-3            # 空间步长 2 mm
Nx, Ny, Nz = int(Lx/dx), int(Ly/dy), int(Lz/dz)

# 时间步长（CFL 条件）
dt = 1.0/(c0*np.sqrt((1/dx**2)+(1/dy**2)+(1/dz**2)))
n_steps = 2000  # 模拟步数，可调

# --- 场变量 ---
Ex = np.zeros((Nx, Ny, Nz))
Ey = np.zeros((Nx, Ny, Nz))
Ez = np.zeros((Nx, Ny, Nz))
Hx = np.zeros((Nx, Ny, Nz))
Hy = np.zeros((Nx, Ny, Nz))
Hz = np.zeros((Nx, Ny, Nz))

# --- 源参数 ---
f0 = 1e9                 # 频率 1 GHz
t0 = 50*dt               # 高斯脉冲延迟
tau = 15*dt              # 脉宽
E0 = 1e4                 # 峰值电场 10 kV/m

# --- 记录探针 ---
src_record = []
probe_record = []

src_z = int(0.05/dz)     # 源在 z=0.05 m
probe_z = int(0.9/dz)    # 探针在 z=0.9 m
ix, iy = Nx//2, Ny//2    # x,y 中心位置

# --- 时间循环 ---
for n in range(n_steps):
    # 更新 H 场
    Hx[:-1,:-1,:-1] -= dt/(mu0*dy) * (Ez[:-1,1:,:-1] - Ez[:-1,:-1,:-1]) \
                      - dt/(mu0*dz) * (Ey[:-1,:-1,1:] - Ey[:-1,:-1,:-1])
    Hy[:-1,:-1,:-1] -= dt/(mu0*dz) * (Ex[:-1,:-1,1:] - Ex[:-1,:-1,:-1]) \
                      - dt/(mu0*dx) * (Ez[1:,:-1,:-1] - Ez[:-1,:-1,:-1])
    Hz[:-1,:-1,:-1] -= dt/(mu0*dx) * (Ey[1:,:-1,:-1] - Ey[:-1,:-1,:-1]) \
                      - dt/(mu0*dy) * (Ex[:-1,1:,:-1] - Ex[:-1,:-1,:-1])

    # 更新 E 场
    Ex[1:,1:,1:] += dt/(eps0*dy) * (Hz[1:,1:,1:] - Hz[1:,:-1,1:]) \
                   - dt/(eps0*dz) * (Hy[1:,1:,1:] - Hy[1:,1:,0:-1])
    Ey[1:,1:,1:] += dt/(eps0*dz) * (Hx[1:,1:,1:] - Hx[1:,1:,0:-1]) \
                   - dt/(eps0*dx) * (Hz[1:,1:,1:] - Hz[0:-1,1:,1:])
    Ez[1:,1:,1:] += dt/(eps0*dx) * (Hy[1:,1:,1:] - Hy[0:-1,1:,1:]) \
                   - dt/(eps0*dy) * (Hx[1:,1:,1:] - Hx[1:,0:-1,1:])

    # 注入源 (高斯调制正弦，沿 z 传播，电场方向取 Ex)
    src_val = E0 * np.exp(-((n*dt - t0)/tau)**2) * np.sin(2*np.pi*f0*n*dt)
    Ex[ix, iy, src_z] += src_val

    # 记录
    src_record.append(src_val)
    probe_record.append(Ex[ix, iy, probe_z])

# --- 绘制结果 ---
time_axis = np.arange(n_steps)*dt*1e9  # ns

plt.figure(figsize=(10,4))
plt.plot(time_axis, src_record, label="Source at z=0.05 m")
plt.plot(time_axis, probe_record, label="Probe at z=0.9 m")
plt.xlabel("Time (ns)")
plt.ylabel("Ex (V/m)")
plt.title("3D FDTD: Source vs Probe signals")
plt.legend()
plt.grid(True)
plt.show()
