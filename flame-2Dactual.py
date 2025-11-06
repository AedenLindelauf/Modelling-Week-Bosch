import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm

# --- Constants (from your original) ---
A, B, C, D = 0.6079, -2.554, 7.31, 1.23

# --- Domain discretization (from your original) ---
x0, xN, dx = 0.0, 0.6, 0.005
y0, yN, dy = 0.0, 0.6, 0.005
t_final, dt = 5.0, 0.01
radius = 0.1
centre = xN / 2

# build grids
x = np.arange(x0, xN + 1e-12, dx)
y = np.arange(y0, yN + 1e-12, dy)
Nx = len(x)
Ny = len(y)
Nt = int(round(t_final / dt)) + 1

# --- Initialize G and phi arrays ---
G = np.zeros((Nx, Ny, Nt))
phi = np.ones((Nx, Ny, Nt)) * 0.5

# --- Flame speed constant used to compute M (steady solution) ---
s_L_constant = 0.0725887

# --- Velocity field ---
eps = 0.3
omega = 20

U = 3 * s_L_constant
u_x = np.zeros((Nx, Ny))
u_y = np.ones((Nx, Ny)) * U
u_y_full = np.zeros((Nx, Ny, Nt))

# compute slope M so that s_L * sqrt(M^2 + 1) = U
# (make sure U/s_L_constant >= 1)
if (U / s_L_constant) <= 1.0:
    raise ValueError("U must be >= s_L for real M.")
M = np.sqrt((U / s_L_constant) ** 2 - 1.0)
print("Using M =", M)

# Set initial field to steady analytic solution: G(x,y) = M*(x - radius) + y
for i in range(Nx):
    for j in range(Ny):
        G[i, j, 0] = M * (abs(x[i] - centre) - radius) + y[j]

# --- Boundary conditions helper:
# bottom (y=0): Dirichlet (inflow)
# left, right, top: outflow / zero-gradient (copy from adjacent interior cell)
def apply_bcs(G_slice):
    # bottom (inflow): Dirichlet
    M = np.sqrt((u_y[0, 0] / s_L_constant) ** 2 - 1.0)
    for ii in range(Nx):
        G_slice[ii, 0] = M * (abs(x[ii] - centre) - radius) + 0.0

    # copy right half from left half
    for jj in range(Ny):
        for ii in range(Nx // 2):
            G_slice[Nx - ii - 1, jj] = G_slice[ii, jj]
            G_slice[Nx - ii - 1, jj] = G_slice[ii, jj]

# Apply BCs at initial time (t=0)
apply_bcs(G[:, :, 0])

# --- CFL check for explicit advection (simple sufficient condition) ---
cfl = dt * (np.abs(u_x).max() / dx + np.abs(u_y).max() / dy)
print("CFL value (dt*(|u_x|/dx + |u_y|/dy)) =", cfl)
if cfl >= 1.0:
    print("Warning: CFL >= 1.0; reduce dt or increase dx/dy for stability.")

# --- Time-stepping: explicit upwind for advection; use previous time for gradients---
for n in range(1, Nt):
    # start from previous time slice
    G_prev = G[:, :, n - 1].copy()
    G_new = G_prev.copy()
    
    t = n * dt
    for j in range(Ny):
        for i in range(Nx):
            u_y[i, j] = U * (1 + eps * np.sin(omega * (t - j*dy)))
            #u_y[i, j] = s_L_constant + (U - s_L_constant) * (2 ** -t)
            u_y_full[i, j, n] = u_y[i, j]
    
    # apply Dirichlet on bottom inflow for this new time slice (makes sure inflow is enforced)
    M = np.sqrt((u_y[0, 0] / s_L_constant) ** 2 - 1.0)
    for ii in range(Nx):
        G_new[ii, 0] = M * (abs(x[ii] - centre) - radius) + 0.0

    # interior points update (i indexes x, j indexes y)
    for j in range(1, Ny - 1):       # avoid top row (Ny-1) because we treat it as outflow
        for i in range(1, Nx - 1):   # avoid left (0) and right (Nx-1)
            # upwind/backward differences because u_x >= 0? here u_x == 0 so dx contribution zero
            # but implement general form:
            if u_x[i, j] >= 0:
                dGdx = (G_prev[i, j] - G_prev[i - 1, j]) / dx
            else:
                dGdx = (G_prev[i + 1, j] - G_prev[i, j]) / dx

            if u_y[i, j] >= 0:
                dGdy = (G_prev[i, j] - G_prev[i, j - 1]) / dy
            else:
                dGdy = (G_prev[i, j + 1] - G_prev[i, j]) / dy

            # local laminar flame speed (using phi at previous time)
            s_L = A * (phi[i, j, n - 1] ** B) * np.exp(-C * (phi[i, j, n - 1] - D) ** 2)

            grad_norm = np.sqrt(dGdx * dGdx + dGdy * dGdy)

            adv = u_x[i, j] * dGdx + u_y[i, j] * dGdy

            # explicit update: G^{n+1} = G^n - dt*(adv - s_L * |grad|)
            G_new[i, j] = G_prev[i, j] - dt * (adv - s_L * grad_norm)

    # set boundaries on bottom which is Dirichlet
    apply_bcs(G_new)

    # save
    G[:, :, n] = G_new

# --- Plot final heatmap ---
final = G[:, :, -1].T  # transpose so rows correspond to y for imshow with origin='lower'
plt.figure(figsize=(6,6))
plt.imshow(final, extent=[x0, xN, y0, yN], origin='lower', aspect='auto')
plt.colorbar(label="G value")
plt.title("Heatmap at final time step")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

# --- Quick diagnostic: print min/max of G over time to confirm steady behavior ---
print("G initial min/max:", G[:, :, 0].min(), G[:, :, 0].max())
print("G final min/max:  ", G[:, :, -1].min(), G[:, :, -1].max())



fig, ax = plt.subplots(figsize=(6,6))

frame0 = u_y_full[:, :, 0].T
im = ax.imshow(frame0, extent=[x[0], x[-1], y[0], y[-1]],
               origin='lower', aspect='auto', cmap="coolwarm", vmin = U * (1 - eps), vmax = U * (1 + eps))

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("u_y value")

ax.set_title("u_y(x,y,t=0)")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    im.set_data(u_y_full[:, :, frame].T)
    ax.set_title(f"u_y(x,y,t={round(frame * dt, 4)})")
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=G.shape[2],
    interval=int(1000 * dt),
    blit=False
)

plt.show()




fig, ax = plt.subplots(figsize=(6,6))

linthresh = 0.1  # values within [-0.1,0.1] change linearly
norm = SymLogNorm(linthresh=linthresh, vmin=np.min(G), vmax=np.max(G), base=10)

frame0 = G[:, :, 0].T
im = ax.imshow(frame0, extent=[x[0], x[-1], y[0], y[-1]],
               origin='lower', aspect='auto', cmap="coolwarm_r", norm=norm)
contour = ax.contour(np.arange(x0, xN + 1e-12, dx), np.arange(y0, yN + 1e-12, dy), G[:, :, 0].T, levels=[0.0], colors='black')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("G value")

ax.set_title("G(x,y,t=0)")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    global contour
    contour.remove()
    im.set_data(G[:, :, frame].T)
    ax.set_title(f"G(x,y,t={round(frame * dt, 4)})")
    contour = ax.contour(np.arange(x0, xN + 1e-12, dx), np.arange(y0, yN + 1e-12, dy), G[:, :, frame].T, levels=[0.0], colors='black')
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=G.shape[2],
    interval=int(1000 * dt),
    blit=False
)

plt.show()
