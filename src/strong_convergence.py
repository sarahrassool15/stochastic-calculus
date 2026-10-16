# strong_convergence.py
# Part 3 
# Compares two ways of simulating GBM, dS = mu S dt + sigma S dW:
#   Euler Maruyama: S_{n+1} = S_n + mu S_n dt + sigma S_n dW
#   Milstein:       same + 0.5 sigma^2 S_n (dW^2 - dt)
# and measures their STRONG error (how far each simulated path ends from the exact path)
# as the step size dt shrinks.
# Theory: strong order 0.5 for Euler-Maruyama and 1 for Milstein.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

S0, mu, sigma, T = 100.0, 0.10, 0.80, 1.0
n_paths = 10_000


def euler_maruyama(S0, mu, sigma, dt, dW):
    # dW has one row per path and one column per time step
    S = np.full(dW.shape[0], S0)
    for k in range(dW.shape[1]):          # loop over TIME (each step needs the previous one)
        S = S + mu * S * dt + sigma * S * dW[:, k]
    return S                              # price at time T on every path


def milstein(S0, mu, sigma, dt, dW):
    S = np.full(dW.shape[0], S0)
    for k in range(dW.shape[1]):
        S = (S + mu * S * dt + sigma * S * dW[:, k]
             + 0.5 * sigma**2 * S * (dW[:, k]**2 - dt))   # the Milstein correction
    return S


# ---------- one fine Brownian path per sample ----------
# NOTE: this is the key trick. I simulate the finest grid ONCE, and build every
# coarser grid by adding up neighbouring fine increments. So every step size
# sees exactly the same Brownian path, and only dt changes.
n_fine = 2**10                            # 1024 steps
dt_fine = T / n_fine
dW_fine = np.sqrt(dt_fine) * rng.standard_normal((n_paths, n_fine))

# exact solution at T only needs W_T, the sum of all the increments
W_T = dW_fine.sum(axis=1)
S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)

step_counts = [2**k for k in range(3, 10)]    # 8, 16, ..., 512 steps
dts, err_em, err_mil = [], [], []

for n in step_counts:
    dt = T / n
    group = n_fine // n                   # how many fine steps make one coarse step
    # reshape each row into n blocks of `group` fine steps, then add up each block
    dW = dW_fine.reshape(n_paths, n, group).sum(axis=2)

    S_em = euler_maruyama(S0, mu, sigma, dt, dW)
    S_mil = milstein(S0, mu, sigma, dt, dW)

    # strong error: average distance between the simulated and exact endpoint
    dts.append(dt)
    err_em.append(np.mean(np.abs(S_em - S_exact)))
    err_mil.append(np.mean(np.abs(S_mil - S_exact)))

dts, err_em, err_mil = map(np.array, (dts, err_em, err_mil))

# convergence order = slope on a log-log plot
slope_em = np.polyfit(np.log(dts), np.log(err_em), 1)[0]
slope_mil = np.polyfit(np.log(dts), np.log(err_mil), 1)[0]

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(dts, err_em, "o-", label=f"Euler-Maruyama (slope {slope_em:.2f})")
ax.loglog(dts, err_mil, "s-", label=f"Milstein (slope {slope_mil:.2f})")
# reference lines with slope 0.5 and 1, anchored at the coarsest point
ax.loglog(dts, err_em[0] * (dts / dts[0])**0.5, "k:", lw=1, label="slope 0.5")
ax.loglog(dts, err_mil[0] * (dts / dts[0])**1.0, "k--", lw=1, label="slope 1")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"Strong error $E|S_T^{\mathrm{scheme}} - S_T^{\mathrm{exact}}|$")
ax.legend()
fig.tight_layout()
fig.savefig("figures/strong_convergence.png", dpi=150)

# ---------- table ----------
print(f"{'steps':>6} {'dt':>9} {'Euler-M error':>14} {'Milstein error':>15}")
for n, dt, e1, e2 in zip(step_counts, dts, err_em, err_mil):
    print(f"{n:>6} {dt:>9.5f} {e1:>14.4f} {e2:>15.4f}")
print(f"\nestimated strong order: Euler-Maruyama {slope_em:.2f} (theory 0.5), "
      f"Milstein {slope_mil:.2f} (theory 1)")