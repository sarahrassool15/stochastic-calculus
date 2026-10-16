# gbm_exact.py
# Part 2 of my stochastic calculus project
# Simulates geometric Brownian motion with its EXACT solution (from Ito's lemma) so 
#     S_t = S_0 * exp((mu - sigma^2/2) t + sigma W_t)
# and checks the Ito correction -sigma^2/2 numerically
# NOTE: I use a Bitcoin like volatility at 80% so the gap between the two is easy to see

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

S0 = 100.0      # starting price
mu = 0.10       # drift: 10% a year
sigma = 0.80    # volatility: 80% a year (roughly Bitcoin)
T = 1.0         # one year
n = 252         # daily steps (trading days)
n_paths = 100_000

dt = T / n
t = np.linspace(0, T, n + 1)

# Brownian paths, exactly as in Part 1
dW = np.sqrt(dt) * rng.standard_normal((n_paths, n))
W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)

# exact GBM solution on every path
# NOTE: no time-stepping error here. This formula is exact at every grid point
# which is why GBM is the perfect test case for the numerical schemes in Part 3
S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# ---------- compare simulation with theory ----------
sim_mean = S.mean(axis=0)            # average across paths, at each time
sim_median = np.median(S, axis=0)
theory_mean = S0 * np.exp(mu * t)
theory_median = S0 * np.exp((mu - 0.5 * sigma**2) * t)

print(f"At T = {T}:")
print(f"  mean   S_T: simulated {sim_mean[-1]:7.2f}   theory {theory_mean[-1]:7.2f}")
print(f"  median S_T: simulated {sim_median[-1]:7.2f}   theory {theory_median[-1]:7.2f}")
print(f"  share of paths that end BELOW S_0: {(S[:, -1] < S0).mean():.1%}")

# ---------- figure: a fan of paths with the mean and median ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, S[:50].T, color="grey", lw=0.5, alpha=0.5)       # 50 sample paths
ax.plot(t, theory_mean, color="C0", lw=2, label=r"mean $S_0e^{\mu t}$")
ax.plot(t, theory_median, color="C3", lw=2,
        label=r"median $S_0e^{(\mu-\sigma^2/2)t}$")
ax.plot(t, sim_mean, "C0--", lw=1)
ax.plot(t, sim_median, "C3--", lw=1)
ax.set_yscale("log")   # NOTE: log scale because GBM paths spread out multiplicatively
ax.set_xlabel("t (years)")
ax.set_ylabel("Price (log scale)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("figures/gbm_mean_vs_median.png", dpi=150)