import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import pandas as pd
import scipy.optimize as sp
import seaborn as sns

# --- Constants ---
A, B, C, D = 0.6079, -2.554, 7.31, 1.23

# --- Domain discretization ---
x0, xN, dx = 0.0, 3.0, 0.1
y0, yN, dy = 0.0, 3.0, 0.1
t_final, dt = 5.0, 0.1
radius = 1.5

x = np.array([x0 + i * dx for i in range(round(xN / dx + 1))])
Nx = len(x)
y = np.array([y0 + i * dy for i in range(round(yN / dy + 1))])
Ny = len(y)
Nt = round(t_final / dt + 1)
N_radius = round(radius / dx)

# --- Velocity field ---
u_x = np.zeros((Nx, Ny, Nt))
u_y = np.ones((Nx, Ny, Nt)) * 0.14
#u[0] = 1.0  # only at boundary

# --- Initial conditions ---
u_constant = 0.14
s_L_constant = 0.0725887


G = np.zeros((Nx, Ny, Nt))
# Initially everything is unburnt
for i in range(Nx):
    for j in range(Ny):
        G[i][j][0] = 1.0

# x-axis boundary condition (constant over time)
#for i in range(Nx):
#    for j in range(Nt):
#        G[i][0][j] = (u_constant**2 - s_L_constant**2)**0.5 / s_L_constant * (i*dx - radius) + 0

# y-axis boundary condition (constant over time)
#for i in range(Ny):
#    for j in range(Nt):
#        G[0][i][j] = (u_constant**2 - s_L_constant**2)**0.5 / s_L_constant * (0 - radius) + i*dy
for i in range(round(radius/dx + 1)):
    for j in range(Nt):
        G[i][0][j] = -1.0


phi = np.ones((Nx, Ny, Nt)) * 0.5
phi_constant = 0.5

# --- Time stepping ---
for t in range(1, Nt):
    for y in range(0, Ny):
        for x in range(1, Nx):

            # phi is constant - NO WORK NEEDED! YAY!
            # Later will implement varying phi

            # Local laminal flame speed
            s_L = A * (phi[x,y,t] ** B) * np.exp(-C * (phi[x,y,t] - D) ** 2)

            # Local value of G, from the G equation
            # Resulting equation (for G[x,y,t]) cannot be solved explicitly, so use fsolve instead
            def diff_eq(z):
                if y != 0:
                    return ((z - G[x,y,t-1])/dt 
                        + u_x[x-1, y, t] * (z - G[x-1,y,t])/dx 
                        + u_y[x, y-1, t] * (z - G[x,y-1,t])/dy 
                        - s_L * (((z - G[x-1,y,t])/dx) ** 2 + ((z - G[x,y-1,t])/dy) ** 2) ** 0.5)
                else:
                    return ((z - G[x,y,t-1])/dt 
                        + u_x[x-1, y, t] * (z - G[x-1,y,t])/dx 
                        + 0 * (z - G[x,y-1,t])/dy 
                        - s_L * (((z - z)/dx) ** 2 + ((z - G[x,y-1,t])/dy) ** 2) ** 0.5)
            G[x,y,t] = sp.fsolve(diff_eq, x0 = G[x-1,y,t])[0]


#print(G)
#print(phi)
#print(s_L)


# --- Results ---
"""df = pd.DataFrame({"x": x, "G(x,15)": [G[i, Nt - 1] for i in range(Nx)], "phi(x,15)": [phi[i, Nt - 1] for i in range(Nx)]})
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
plt.show()"""
final_heatmap_data = [[G[i,j,1] for i in range(Nx)] for j in range(Ny)]
final_heatmap_data = np.array(final_heatmap_data)
#heatmap = sns.heatmap(final_heatmap_data, norm = mcolors.LogNorm())
plt.imshow(final_heatmap_data, extent = [x0, xN-x0, y0, yN-y0], origin = 'lower')
plt.colorbar(label = "G value")
#colorbar = heatmap.collections[0].colorbar
#colorbar.set_label('Log Scale Colorbar')
#colorbar.ax.set_ylabel('Logarithmic Scale', rotation=270, labelpad=20)
plt.title("Heatmap at final time step")

plt.show()
