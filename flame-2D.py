import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp

# --- Constants ---
A, B, C, D = 0.6079, -2.554, 7.31, 1.23

# --- Domain discretization ---
x0, xN, dx = 0.0, 5.0, 0.1
y0, yN, dy = 0.0, 5.0, 0.1
t_final, dt = 15.0, 0.1
radius = 1.5

x = np.array([x0 + i * dx for i in range(round(xN / dx + 1))])
Nx = len(x)
y = np.array([y0 + i * dy for i in range(round(yN / dy + 1))])
Ny = len(y)
Nt = round(t_final / dt + 1)
N_radius = round(radius / dx)

# --- Velocity field ---
u = np.zeros((Nx, Ny, Nt, 2))
#u[0] = 1.0  # only at boundary

# --- Initial conditions ---
u_constant = 2.0
s_L_constant = 1.0


G = np.zeros((Nx, Ny, Nt))
# Initially everything is unburnt
for i in range(Nx):
    for j in range(Ny):
        G[i][j][0] = 1.0

# x-axis boundary condition (constant over time)
for i in range(Nx):
    for j in range(Nt):
        G[i][0][j] = (u_constant**2 - s_L_constant**2)**0.5 / s_L_constant * (i*Nx - radius)

# y-axis boundary condition (constant over time)
for i in range(Ny):
    for j in range(Nt):
        G[0][i][j] = (u_constant**2 - s_L_constant**2)**0.5 / s_L_constant * (i*Ny - radius) + i*Ny


phi = np.ones((Nx, Ny, Nt)) * 1.0

# --- Time stepping ---
for t in range(1, Nt):
    for y in range(1, Ny):
        for x in range(1, Nx):

            # phi is constant - NO WORK NEEDED! YAY!
            s_L = A * (phi[x,y,t] ** B) * np.exp(-C * (phi[x,y,t] - D) ** 2)
            def diff_eq(z):
                return (z - G[x,y,t-1])/dt + u[]



for t in range(1, Nt):
    for y in range(1, Nx):
        #G_old = G.copy()
        #phi_old = phi.copy()

        phi[y, t] = (dx * phi[y, t-1] + dt * u[y-1] * phi[y-1, t]) / (dx + dt * u[y-1])
        s_L = A * (phi[y,t] ** B) * np.exp(-C * (phi[y,t] - D) ** 2)
        def diff_eq(z):
            return (z - G[y, t-1])/dt + u[y-1] * (z - G[y-1, t])/dx - s_L * abs(z - G[y-1, t])/dx
        G[y, t] = sp.fsolve(diff_eq, x0 = G[y-1, t])[0]


print(G)
print(phi)
print(s_L)
"""for n in range(Nt):
    G_old = G.copy()
    phi_old = phi.copy()
    
    # update phi using forward Euler + upwind
    for i in range(1, Nx):
        phix = (phi_old[i] - phi_old[i-1]) / dx
        phi[i] = phi_old[i] - dt * u[i] * phix
    phi[0] = 0.0  # boundary condition

    # compute s_L(phi)
    
    
    # update G using forward Euler
    for i in range(1, Nx):
        Gx = (G_old[i] - G_old[i-1]) / dx
        G[i] = G_old[i] + dt * (-u[i] * Gx + s_L[i] * abs(Gx))
    G[0] = -1.0  # boundary condition"""

# --- Results ---
df = pd.DataFrame({"x": x, "G(x,15)": [G[i, Nt - 1] for i in range(Nx)], "phi(x,15)": [phi[i, Nt - 1] for i in range(Nx)]})
print(df.head(), "\n...\n", df.tail())

# --- Plot results ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x, G)
plt.title(f"G(x, t={t_final})")
plt.xlabel("x"); plt.ylabel("G")

plt.subplot(1,2,2)
plt.plot(x, phi)
plt.title(f"phi(x, t={t_final})")
plt.xlabel("x"); plt.ylabel("phi")

plt.tight_layout()
plt.show()
