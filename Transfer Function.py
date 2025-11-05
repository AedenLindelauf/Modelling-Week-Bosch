from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def G2(x, y, t, omega): # updated G function using 
    # Precompute geometry
    X = x*np.cos(alpha - np.pi/2) - y*np.sin(alpha - np.pi/2) - U_bar*t
    Y = x*np.sin(alpha - np.pi/2) + y*np.cos(alpha - np.pi/2)
    
    # Effective wavenumber
    k = 0.5 * omega * v_Bar * (np.cos(alpha) - 1/np.cos(alpha))
    
    # Reference X0 (the inlet, moving with -U_bar*t)
    X0 = -U_bar * t
    
    # xi with inlet correction so xi=0 at X=-U_bar*t
    xi = (v1 * np.sin(alpha) / U_bar) * ((np.sin(k*X) - np.sin(k*X0)) / k)
    
    return Y - xi



def calculate_q(time, omega):
    Z = G2(X, Y, time, omega)
    
    # Compute gradients
    dGdx, dGdy = np.gradient(Z, dx, dy)
    gradG = np.sqrt(dGdx**2 + dGdy**2)
    
    # Approximate delta function
    delta_G = (1/(epsilon*np.sqrt(np.pi))) * np.exp(-(Z**2)/(epsilon**2))
    
    # Numerical integral
    return ( 2 * np.pi * rho * h * np.sum(s_L * gradG * delta_G * dx * dy) )

# Meshgrid
x = np.linspace(0, 1, 200)
y = np.linspace(0, 2, 200)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)


# Parameters
rho = 1.0
h = 1.0
#s_L = 1.0
epsilon = 0.01  # delta function approximation

t = np.linspace(0, 100, 10000)

# Number of samples
N = len(t)

# Sample frequency
Fs = 10000
dt = 1 / Fs

tf = fftfreq(N, dt)

# u input velocity
#U_bar = 1

Q_bar = 1
St = 1

s_L = 1


Sl = 0.31      # 0.31 for Methane, and 2.88 m/s
v_Bar =  0.5      # mean upward velocity in the lab frame
v1 = 0.01     # magnitude of velocity perturbation
alpha = np.arcsin(Sl/v_Bar)

U_bar = v_Bar*np.cos(alpha)

eps_a = v1


omegas = np.linspace(1,100)

tf = tf[:len(tf)//2]

f_max = []
v_max = []

for omega in omegas:
    q = [0] * len(t)

    u = lambda t : eps_a * U_bar * np.sin(St * t)
    uf = fft(u(t))

    for i in range(len(t)):
        q[i] = calculate_q(t[i], omega)
        
    qf = fft(q)

    # 0 \leq index < N
    FTF_u = lambda index : qf[index] * U_bar / (uf[index] * Q_bar)


    res = [FTF_u(i) for i in range(len(uf))]

    res = res[:len(res)//2]
    
    index_max = np.argmax( np.abs(res[1:]) / N )
    f_max.append( tf[index_max] )
    v_max.append( np.abs(res[index_max]) / N )


plt.scatter(f_max, v_max)
plt.show()

