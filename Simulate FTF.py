import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from numba import njit, prange

# ---- User settings ----
N_CPUS = 32                # Number of CPUs for multithreading
SAVE_EVERY = 20            # Save every N frequencies
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

# Sampling parameters change for faster simulation or accuracy
n_periods = 2000
n_samples = 10000

# Mesh grid setup
nx = 200
ny = 200
x = np.linspace(0, 1, nx)

# Frequency range of perturbation (scales differently then the output frequency) 
omegas = np.linspace(104,2500,5000)

# Constants should not matter as the FTF is normalized by Q_bar
rho = 1.0
h = 1.0
epsilon = 0.01

# ----------------------------
# Vectorized + numba-accelerated inner kernel
# ----------------------------
@njit(fastmath=True) # removed parallel=True, using only one cpu per omega
def compute_q_numba(X_base, Y_base, dx, dy, s_L, v_Bar, v1, alpha, U_bar,
                    rho, h, epsilon, t, omega):
    """Fast JIT-compiled function for computing q(t, ?)."""
    k = 0.5 * omega * v_Bar * (np.cos(alpha) - 1.0 / np.cos(alpha))
    X0 = U_bar * t
    nx, ny = X_base.shape
    total = 0.0

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    coeff = (v1 * sin_alpha / U_bar)

    for i in prange(nx):
        for j in range(ny):
            X = X_base[i, j] + U_bar * t
            Y = Y_base[i, j]

            # compute xi
            if abs(k) < 1e-12:
                xi = coeff * (X - X0)
            else:
                xi = coeff * (np.sin(k * X) - np.sin(k * X0)) / k

            Z = Y - xi
            cos_kX = np.cos(k * X)

            dZdx = -cos_alpha - (v1 * (sin_alpha ** 2) / U_bar) * cos_kX
            dZdy = sin_alpha - (v1 * (sin_alpha * cos_alpha) / U_bar) * cos_kX

            gradG = np.sqrt(dZdx * dZdx + dZdy * dZdy)
            delta_G = (1.0 / (epsilon * np.sqrt(np.pi))) * np.exp(-(Z ** 2) / (epsilon ** 2))

            total += s_L * gradG * delta_G

    integral = total * dx * dy
    q_val = 2.0 * np.pi * rho * h * integral
    return q_val

def process_single_omega(omega, params):
    """Compute FTF for one omega value."""
    X_base, Y_base, dx, dy, s_L, v_Bar, v1, alpha, U_bar = params
    eps_a = v1

    St = omega # / U_bar

    # Make time array with integer number of input periods to reduce leakage
    T_period = 2 * np.pi / St
    t = np.linspace(0, n_periods * T_period, n_samples)
    N = len(t)
    dt = t[1] - t[0]

    tf = fftfreq(N, dt)[:N//2]

    # Input velocity perturbation
    u_t = eps_a * U_bar* np.sin(St * t)

    # Compute q(t)
    q = np.empty(N, dtype=np.float64)
    for i in range(N):
        q[i] = compute_q_numba(X_base, Y_base, dx, dy, s_L, v_Bar, v1, alpha, U_bar,
                               rho, h, epsilon, t[i], omega)

    Q_bar = np.mean(q)
    
    w = np.hanning(N)
    q_win = (q - Q_bar) * w
    u_win = u_t * w

    Qf = np.fft.fft(q_win)[:N//2]
    Uf = np.fft.fft(u_win)[:N//2]

    # equivalent to multiplying by 2/sum(w)
    norm_factor = 2.0 / np.sum(w)
    Qf = Qf * norm_factor
    Uf = Uf * norm_factor

    # protect denom
    denom = Uf.copy()
    denom[np.abs(denom) < 1e-20] = 1e-20

    # FTF (use these normalized spectra)
    FTF_u = Qf * U_bar / (denom * Q_bar)
    FTF = FTF_u /N  # <- no extra / N?

    mags = np.abs(FTF_u)
    idx_max = np.argmax(mags[1:]) + 1 if len(mags) > 1 else 0 #change 1 to maybe 100 if many times omega ~0
    max_out = FTF[idx_max]
    max_freq = tf[idx_max]
    
    return omega, FTF, max_out, max_freq

# ----------------------------
# Main gas-processing routine
# ----------------------------
def run_for_gas(gas_label, s_L_value, v1_value, save_prefix, n_cpus=N_CPUS):
    print(f"Starting run for {gas_label} with s_L={s_L_value}, v1={v1_value}")

    s_L = s_L_value
    v_Bar = 4.0 * s_L
    alpha = np.arcsin(s_L / v_Bar)
    U_bar = v_Bar * np.cos(alpha)
    eps_a = v1_value

    # Build mesh grid
    y_max = float(np.floor(1.0 / np.tan(alpha)) + 0.2)
    y = np.linspace(0, y_max, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X_grid, Y_grid = np.meshgrid(x, y)

    # Precompute coordinate transforms
    X_base = X_grid * np.sin(alpha) + Y_grid * np.cos(alpha)
    Y_base = -X_grid * np.cos(alpha) + Y_grid * np.sin(alpha)

    params = (X_base, Y_base, dx, dy, s_L, v_Bar, eps_a, alpha, U_bar)

    # Results containers
    half = n_samples // 2
    ftfs = np.zeros((len(omegas), half), dtype=np.complex128)
    trans = np.zeros((len(omegas),3), dtype=np.complex128)

    with Pool(processes=n_cpus) as pool:
        worker = partial(process_single_omega, params=params)
        for j, res in enumerate(pool.imap(worker, omegas)):
            omega_val, FTF, max_out, max_freq = res  
            ftfs[j, :] = FTF
            trans[j, :] = np.array([omega_val, max_freq, max_out])


    
            if j % SAVE_EVERY == 0:
                np.save(OUTDIR / f"{save_prefix}_ftfs_partial.npy", ftfs) 
                np.save(OUTDIR / f"{save_prefix}_trans.npy", trans)

    # Final save
    np.save(OUTDIR / f"{save_prefix}_ftfs.npy", ftfs)
    np.save(OUTDIR / f"{save_prefix}_trans.npy",trans)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(trans[:,1].real, np.abs(trans[:,2]), s=20)
    plt.yscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (or velocity amplitude)")
    plt.title(f"Frequency Response ({gas_label})")
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{save_prefix}_frequency_response.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Finished {gas_label}. Results saved under {OUTDIR}")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    # Methane case
    run_for_gas(
        gas_label="H2",
        s_L_value=1.6,   # flame speed (m/s) 0.25 methane, 1.6 H2
        v1_value=0.01,    # perturbation amplitude
        save_prefix="H2",
        n_cpus=N_CPUS
    )
