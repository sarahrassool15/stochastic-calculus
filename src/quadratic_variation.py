# quadratic_variation.py
# Part 1 of my project L3
# Checks numerically that the quadratic variation of Brownian motion on [0, T] is T:
#     sum of (dW)^2  ->  T   as dt -> 0
# For a smooth function this sum would go to 0. For brownian motion it doesn't
# and that's the reason Ito's lemma has an extra term.

import numpy as np
import matplotlib.pyplot as plt

# NOTE: fixed seed so every run gives exactly the same results
rng = np.random.default_rng(42)

T = 1.0  # time horizon

# ---------- fig 1: a few paths and their running quadratic variation ----------
n = 10_000          # number of steps
dt = T / n
n_paths = 5

# Brownian increments: each one is N(0, dt), so its typical size is sqrt(dt)
# NOTE: sqrt(dt), not dt. This is the whole reason nrownian motion is so jagged
dW = np.sqrt(dt) * rng.standard_normal((n_paths, n))   # one row per path   #Important 
# the path is the running total of the increments, starting at 0
W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)   

# running quadratic variation: running total of the SQUARED increments
QV = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW**2, axis=1)], axis=1)   

t = np.linspace(0, T, n + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for i in range(n_paths):
    axes[0].plot(t, W[i], lw=0.8)
    axes[1].plot(t, QV[i], lw=1)
axes[1].plot(t, t, "k--", lw=1.5, label="y = t")
axes[0].set_title("Brownian paths")
axes[1].set_title("Running sum of squared increments")
for ax in axes:
    ax.set_xlabel("t")
axes[1].legend()
fig.tight_layout()
fig.savefig("figures/qv_paths.png", dpi=150)

# ---------- fig 2 quadratic variation concentrates at T as dt shrinks ----------
# Theory is that the sum of (dW)^2 has mean T and variance 2*T*dt
#   (each (dW)^2 = dt * Z^2, with E[Z^2] = 1 and Var(Z^2) = 2, and there are T/dt of them)
# So the spread should shrink like sqrt(2*T*dt): a slope of 1/2 

n_sims = 1_000                       # how many independent paths per step size
step_counts = [10, 100, 1_000, 10_000]

dts, means, stds = [], [], []
for n in step_counts:
    dt = T / n
    dW = np.sqrt(dt) * rng.standard_normal((n_sims, n))
    qv = (dW**2).sum(axis=1)         # one quadratic variation 
    dts.append(dt)
    means.append(qv.mean())
    stds.append(qv.std())

dts = np.array(dts)
stds = np.array(stds)
theory = np.sqrt(2 * T * dts)

# estimated slope of log(std) against log(dt) -> should be about 0.5
slope = np.polyfit(np.log(dts), np.log(stds), 1)[0]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(dts, stds, "o-", label=f"simulated (slope {slope:.2f})")
ax.loglog(dts, theory, "k--", label=r"theory $\sqrt{2T\Delta t}$")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel("Std. dev. of quadratic variation")
ax.legend()
fig.tight_layout()
fig.savefig("figures/qv_convergence.png", dpi=150)

# ---------- Table ----------
print(f"{'steps':>7} {'dt':>8} {'mean QV':>9} {'std':>8} {'theory std':>11}")
for n, dt, m, s, th in zip(step_counts, dts, means, stds, theory):
    print(f"{n:>7} {dt:>8.4f} {m:>9.4f} {s:>8.4f} {th:>11.4f}")
print(f"\nestimated slope: {slope:.3f} (theory: 0.5)")