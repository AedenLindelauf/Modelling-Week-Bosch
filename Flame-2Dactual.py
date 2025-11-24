import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm

# --- Fuel gas ---
# M = methane, H = hydrogen
gas = "M"

if gas == "M":
    def get_s_L(phi):
        A, B, C, D = 0.6079, -2.554, 7.31, 1.23
        return A * (phi ** B) * np.exp(-C * (phi - D) ** 2)
elif gas == "H":
    def get_s_L(phi):
        A, B, C, D = -1.11019, 4.65167, -1.44347, 0.04868
        return A + B * phi + C * phi ** 2 + D * phi ** 3
else:
    print(">:(")



# --- Domain discretization ---
x0, xN, dx = 0.0, 0.5, 0.01
y0, yN, dy = 0.0, 0.5, 0.01
t_final, dt = 5.0, 0.005
radii = [0.15] 
centres = [0.25]

# --- Build grids ---
x = np.arange(x0, xN + 1e-12, dx)
y = np.arange(y0, yN + 1e-12, dy)
Nx = len(x)
Ny = len(y)
Nt = int(round(t_final / dt)) + 1
G = np.zeros((Nx, Ny, Nt))

# --- Constants ---
omega_u = 10
omega_phi = 10
K = 5
eps_a = 0.1
eps_phi = -0.1

# --- Equivalence ratio ---
PHI = 0.5
phi = np.ones((Nx, Ny, Nt)) * PHI
for n in range(Nt):
    t = n * dt
    for j in range(Ny):
        for i in range(Nx):
            y_ = j * dy
            phi[i, j, n] = PHI * (1 + eps_phi * np.sin(omega_phi * (t - y_)))

# --- Flame speed constant used to compute M (steady solution) ---
s_L_constant = get_s_L(PHI)

# --- Velocity field ---
U = 2 * s_L_constant
u_x = np.zeros((Nx, Ny))
u_y = np.ones((Nx, Ny)) * U
u_y_full = np.zeros((Nx, Ny, Nt))
for i in range(Nx):
    for j in range(Ny):
        y_ = j * dy
        u_y[i, j] = U * (1 + eps_a * np.sin(omega_u * (0 - K * y_)))
        u_y_full[i, j, 0] = U * (1 + eps_a * np.sin(omega_u * (0 - K * y_)))

# compute slope M so that s_L * sqrt(M^2 + 1) = U
# (make sure U/s_L_constant >= 1)
if (U / s_L_constant) <= 1.0:
    raise ValueError("U must be >= s_L for real M.")
M = np.sqrt((U / s_L_constant) ** 2 - 1.0)
print("Using M =", M)

# Set initial field to steady analytic solution: G(x,y) = M*(x - radius) + y
for i in range(Nx):
    for j in range(Ny):
        G[i, j, 0] = 1.0
for i in range(Nx):
    possible_minimums = [(M * (abs(i*dx - c) - r)) for c in centres for r in radii]
    G[i, 0, 0] = min(possible_minimums)
    #for j in range(Ny):
        #G[i, j, 0] = M * (abs(x[i] - centre) - radius) + y[j]
        #G[i,j,0] = min(possible_minimums) + y[j]

# --- Boundary conditions helper:
# bottom (y=0): Dirichlet (inflow)
# left, right, top: outflow / zero-gradient (copy from adjacent interior cell)
def apply_bcs(G_slice):
    # bottom (inflow): Dirichlet
    M = np.sqrt((u_y[0, 0] / s_L_constant) ** 2 - 1.0)
    #for ii in range(Nx):
        #G_slice[ii, 0] = M * (abs(x[ii] - centre) - radius) + 0.0
    for ii in range(Nx):
        possible_minimums = [M * (abs(ii*dx - c) - r) for c in centres for r in radii]
        G_slice[ii, 0] = min(possible_minimums)

    # copy right half from left half
    """for k in range(len(centres)):
        for jj in range(Ny):
            for ii in range(round((radii[k])/dx)):
                G_slice[round(centres[k]/dx) + 1 + ii, jj] = G_slice[round(centres[k]/dx) - ii, jj]
                #G_slice[Nx - ii - 1, jj] = G_slice[ii, jj]"""

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
            y_ = j * dy
            u_y[i, j] = U * (1 + eps_a * np.sin(omega_u * (t - K * y_)))
            #u_y[i, j] = s_L_constant + (U - s_L_constant) * (2 ** -t)
            u_y_full[i, j, n] = u_y[i, j]
    
    # apply Dirichlet on bottom inflow for this new time slice (makes sure inflow is enforced)
    M = np.sqrt((u_y[0, 0] / s_L_constant) ** 2 - 1.0)
    #for ii in range(Nx):
        #G_new[ii, 0] = M * (abs(x[ii] - centre) - radius) + 0.0
    for ii in range(Nx):
        possible_minimums = [M * abs(ii*dx - c) - r for c in centres for r in radii]
        G_new[ii, 0] = min(possible_minimums)

    # interior points update (i indexes x, j indexes y)
    for j in range(1, Ny):       # avoid top row (Ny-1) because we treat it as outflow
        for i in range(1, Nx - 1):   # avoid left (0) and right (Nx-1)
            # upwind/backward differences because u_x >= 0? here u_x == 0 so dx contribution zero
            # but implement general form:
            x_ = i * dx
            nearest_flame_centre = 1e10
            for c in centres:
                if abs(x_ - c) < nearest_flame_centre:
                    nearest_flame_centre = x_ - c

            if nearest_flame_centre < 0:
                dGdx = (G_prev[i, j] - G_prev[i - 1, j]) / dx
            else:
                dGdx = (G_prev[i + 1, j] - G_prev[i, j]) / dx
            dGdy = (G_prev[i, j] - G_prev[i, j - 1]) / dy

            # local laminar flame speed (using phi at previous time)
            s_L = get_s_L(phi[i, j, n - 1])

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


def plot_scalar_field(data, name):
    fig, ax = plt.subplots(figsize=(6,6))

    frame0 = data[:, :, 0].T
    im = ax.imshow(frame0, extent=[x[0], x[-1], y[0], y[-1]],
                origin='lower', aspect='auto', cmap="coolwarm", vmin=np.min(data), vmax=np.max(data))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{name} value")

    ax.set_title(f"{name}(x,y,t=0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        im.set_data(data[:, :, frame].T)
        ax.set_title(f"{name}(x,y,t={round(frame * dt, 4)})")
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=G.shape[2],
        interval=int(1000 * dt),
        blit=False
    )

    plt.show()

plot_scalar_field(u_y_full, "u_y")
plot_scalar_field(phi, "phi")



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
    ax.set_title("G(x,y,t=%.3f" % (frame * dt))
    contour = ax.contour(np.arange(x0, xN + 1e-12, dx), np.arange(y0, yN + 1e-12, dy), G[:, :, frame].T, levels=[0.0], colors='black')
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=G.shape[2],
    interval=int(1000 * dt),
    blit=False
)
#ani.save("animation.gif", writer="pillow", fps=int(1/dt))

plt.show()
