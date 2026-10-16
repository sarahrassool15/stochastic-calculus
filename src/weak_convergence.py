# weak_convergence.py
# Part 4 
# Measures the WEAK error of Euler-Maruyama and Milstein for GBM:
#     weak error = | E[f(S_T scheme)] - E[f(S_T exact)] |
# i.e. is the AVERAGE right, not each path. This is what matters for pricing,
# because a price is an expectation.
# Theory: weak order 1 for BOTH schemes

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

S0, mu, sigma, T = 100.0, 0.10, 0.80, 1.0
n_paths = 500_000        # NOTE: many more paths than Part 3, because weak errors are tiny
                         # and Monte Carlo noise shrinks only like 1/sqrt(n_paths)


def euler_maruyama(S0, mu, sigma, dt, dW):
    S = np.full(dW.shape[0], S0)
    for k in range(dW.shape[1]):
        S = S + mu * S * dt + sigma * S * dW[:, k]
    return S


def milstein(S0, mu, sigma, dt, dW):
    S = np.full(dW.shape[0], S0)
    for k in range(dW.shape[1]):
        S = (S + mu * S * dt + sigma * S * dW[:, k]
             + 0.5 * sigma**2 * S * (dW[:, k]**2 - dt))
    return S


# same coupling trick as Part 3: one fine path, coarser grids built by summing
n_fine = 32
dW_fine = np.sqrt(T / n_fine) * rng.standard_normal((n_paths, n_fine))
S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dW_fine.sum(axis=1))

step_counts = [1, 2, 4, 8, 16, 32]
dts, mc_em, mc_mil, th_em, th_mil = [], [], [], [], []

for n in step_counts:
    dt = T / n
    dW = dW_fine.reshape(n_paths, n, n_fine // n).sum(axis=2)
    S_em = euler_maruyama(S0, mu, sigma, dt, dW)
    S_mil = milstein(S0, mu, sigma, dt, dW)

    # Monte Carlo estimate of the weak error.
    # NOTE: the variance reduction trick. Instead of mean(S_em^2) - (exact formula),
    # I average the DIFFERENCE S_em^2 - S_exact^2 on the SAME paths.
    # It has the same expected value, but most of the randomness cancels.
    dts.append(dt)
    mc_em.append(abs(np.mean(S_em**2 - S_exact**2)))
    mc_mil.append(abs(np.mean(S_mil**2 - S_exact**2)))

    # Theory, to check the simulation. Taking expectations one step at a time
    exact_m2 = S0**2 * np.exp((2 * mu + sigma**2) * T)
    th_em.append(abs(S0**2 * ((1 + mu*dt)**2 + sigma**2*dt)**n - exact_m2))
    th_mil.append(abs(S0**2 * ((1 + mu*dt)**2 + sigma**2*dt + 0.5*sigma**4*dt**2)**n - exact_m2))

dts, mc_em, mc_mil, th_em, th_mil = map(np.array, (dts, mc_em, mc_mil, th_em, th_mil))

# fit slopes on the 4 finest step sizes, where the error is in its asymptotic regime
slope_em = np.polyfit(np.log(dts[2:]), np.log(mc_em[2:]), 1)[0]
slope_mil = np.polyfit(np.log(dts[2:]), np.log(mc_mil[2:]), 1)[0]

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(dts, mc_em, "o", color="C0", label=f"Euler-Maruyama, simulated (slope {slope_em:.2f})")
ax.loglog(dts, th_em, "-", color="C0", lw=1, label="Euler-Maruyama, exact")
ax.loglog(dts, mc_mil, "s", color="C1", label=f"Milstein, simulated (slope {slope_mil:.2f})")
ax.loglog(dts, th_mil, "-", color="C1", lw=1, label="Milstein, exact")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"Weak error $|E[S_T^2]_{\mathrm{scheme}} - E[S_T^2]|$")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/weak_convergence.png", dpi=150)

# ---------- table ----------
print(f"{'steps':>6} {'dt':>7} {'EM sim':>9} {'EM exact':>9} {'Mil sim':>9} {'Mil exact':>9}")
for n, dt, a, b, c, d in zip(step_counts, dts, mc_em, th_em, mc_mil, th_mil):
    print(f"{n:>6} {dt:>7.4f} {a:>9.1f} {b:>9.1f} {c:>9.1f} {d:>9.1f}")
print(f"\nestimated weak order: Euler-Maruyama {slope_em:.2f}, Milstein {slope_mil:.2f} (theory 1 for both)")