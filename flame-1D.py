import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp

# --- Constants ---
A, B, C, D = 0.6079, -2.554, 7.31, 1.23

# --- Domain discretization ---
x0, xN, dx = 0.0, 5.0, 0.1
t_final, dt = 15.0, 0.1

x = np.array([x0 + i * dx for i in range(round(xN / dx + 1))])
Nx = len(x)
Nt = round(t_final / dt + 1)

# --- Velocity field ---
u = np.ones(Nx)
#u[0] = 1.0  # only at boundary

# --- Initial conditions ---
G = np.ones((Nx, Nt))
#for i in range(Nx):
 #   G[i][0] = -1.0
for i in range(Nt):
    G[0][i] = -1.0

phi = np.ones((Nx, Nt)) * 0.5
for i in range(Nt):
    phi[0][i] = 0.5

phi_constant = 0.5
s_L_constant = A * (phi_constant ** B) * np.exp(-C * (phi_constant - D) ** 2)
u = np.ones(Nx) * -0.6 * s_L_constant
for i in range(10):
    u[i] = 0.5

# --- Time stepping ---
for t in range(1, Nt):
    for y in range(1, Nx):
        #G_old = G.copy()
        #phi_old = phi.copy()

        # Formula to update local equivalence ratio - formula found explicitly
        # (because the differential equation for it is linear)
        phi[y, t] = (dx * phi[y, t-1] + dt * u[y-1] * phi[y-1, t]) / (dx + dt * u[y-1])

        # Local laminal flame speed
        #s_L = A * (phi[y,t] ** B) * np.exp(-C * (phi[y,t] - D) ** 2)
        s_L = s_L_constant

        # Local value of G, from the G equation.
        # Resulting equation (for G[x,t]) cannot be solved explicitly, so use fsolve instead
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
