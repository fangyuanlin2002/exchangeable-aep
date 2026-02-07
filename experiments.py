import os
import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def H_bin(t):
    """Binary entropy in bits, with safe clipping."""
    t = np.clip(t, 1e-12, 1 - 1e-12)
    return -(t * np.log2(t) + (1 - t) * np.log2(1 - t))

def info_density_theta(theta, x):
    """
    iota_n(theta; x^n) = -(1/n) log2 prod theta^{x_i} (1-theta)^{1-x_i}.
    """
    n = len(x)
    k = int(np.sum(x))
    theta = float(np.clip(theta, 1e-12, 1 - 1e-12))
    return -(k * np.log2(theta) + (n - k) * np.log2(1 - theta)) / n

def log_binom(n, k):
    """log binomial coefficient in natural logs via lgamma."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def ensure_figdir():
    os.makedirs("figures", exist_ok=True)

# -----------------------------
# Experiment 1: Beta--Bernoulli conditional AEP scatter
# -----------------------------

def fig_beta_bern_scatter(alpha=2.0, beta=2.0, n=400, trials=800, seed=0):
    rng = np.random.default_rng(seed)
    thetas = rng.beta(alpha, beta, size=trials)
    iotas = np.zeros(trials)
    ents = H_bin(thetas)

    for t in range(trials):
        x = rng.binomial(1, thetas[t], size=n)
        iotas[t] = info_density_theta(thetas[t], x)

    plt.figure(figsize=(8, 5))
    plt.scatter(ents, iotas, s=10)
    lo = min(np.min(ents), np.min(iotas))
    hi = max(np.max(ents), np.max(iotas))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(r"$H_{\mathrm{bin}}(\Theta)$ (bits)")
    plt.ylabel(r"$\iota_n(\Theta;X_1^n)$ (bits)")
    plt.title(f"Beta--Bernoulli conditional AEP (n={n}, trials={trials})")
    plt.tight_layout()
    plt.savefig("figures/beta_bern_scatter.pdf")
    plt.close()

# -----------------------------
# Experiment 2: Rigidity probability spread inside a deterministic high-mass set
# -----------------------------

def mixture_sequence_prob(x, p1, p2, w=0.5):
    """Mixture of two Bernoulli product measures in bits: P(x)=w P_p1(x)+(1-w) P_p2(x)."""
    n = len(x)
    k = int(np.sum(x))
    # log2 P_p(x) = k log2 p + (n-k) log2(1-p)
    def log2P(p):
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        return k * np.log2(p) + (n - k) * np.log2(1 - p)
    a = log2P(p1)
    b = log2P(p2)
    # stable log-sum-exp base2
    m = max(a, b)
    return 2 ** (m) * (w * 2 ** (a - m) + (1 - w) * 2 ** (b - m))

def fig_rigidity_prob_spread(p1=0.15, p2=0.50, w=0.5, ns=(50, 100, 150, 200, 300), seed=1):
    """
    Deterministic A_n: sequences whose empirical mean in [p1-0.02, p2+0.02].
    Show max/min mixture probability among sampled sequences from that set grows ~ exp(c n).
    """
    rng = np.random.default_rng(seed)
    spreads = []
    for n in ns:
        # sample many sequences from the mixture, keep those in the deterministic set
        M = 40000
        thetas = rng.choice([p1, p2], size=M, p=[w, 1-w])
        X = rng.binomial(1, thetas[:, None], size=(M, n))
        phat = X.mean(axis=1)
        mask = (phat >= (p1 - 0.02)) & (phat <= (p2 + 0.02))
        X_keep = X[mask]
        if len(X_keep) < 50:
            spreads.append(np.nan)
            continue
        # compute mixture probs for a subsample (to control runtime)
        idx = rng.choice(len(X_keep), size=min(3000, len(X_keep)), replace=False)
        probs = np.array([mixture_sequence_prob(X_keep[i], p1, p2, w=w) for i in idx])
        spread = np.max(probs) / np.min(probs)
        spreads.append(spread)

    plt.figure(figsize=(8, 5))
    plt.plot(list(ns), np.log2(spreads), marker="o")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\log_2(\max_{x\in A_n} P(x) / \min_{x\in A_n} P(x))$")
    plt.title("Rigidity: exponential probability spread inside deterministic high-mass sets")
    plt.tight_layout()
    plt.savefig("figures/rigidity_prob_spread.pdf")
    plt.close()

# -----------------------------
# Experiment 3: finite-population correction
# -----------------------------

def finite_population_logprob_ordered(x, N_counts):
    """
    Probability of an *ordered* sample x^n when sampling without replacement uniformly
    from a population with counts N_counts over symbols {0,...,m-1}.
    Exact via sequential drawing:
      P(x^n) = prod_{i=1}^n (remaining_count[x_i] / (N - i + 1)).
    Return - (1/n) log2 P(x^n).
    """
    remaining = np.array(N_counts, dtype=int).copy()
    N = int(np.sum(remaining))
    n = len(x)
    log2p = 0.0
    for i, xi in enumerate(x, start=1):
        denom = N - i + 1
        num = remaining[xi]
        # assume xi is feasible
        log2p += np.log2(num) - np.log2(denom)
        remaining[xi] -= 1
    return -log2p / n

def fig_finite_population_correction(p=(0.2, 0.5, 0.3), N=4000, ns=(50, 80, 120, 180, 260, 360, 520), trials=300, seed=2):
    rng = np.random.default_rng(seed)
    p = np.array(p, dtype=float)
    m = len(p)
    # build population counts
    N_counts = np.floor(N * p).astype(int)
    # fix rounding
    N_counts[0] += N - int(np.sum(N_counts))
    H_p = -np.sum(p * np.log2(np.clip(p, 1e-12, 1.0)))

    ys_raw = []
    ys_corr = []
    xs = []

    # build explicit population array
    population = np.concatenate([np.full(N_counts[j], j, dtype=int) for j in range(m)])

    for n in ns:
        vals = []
        for _ in range(trials):
            sample = rng.choice(population, size=n, replace=False)
            vals.append(finite_population_logprob_ordered(sample, N_counts))
        vals = np.array(vals)
        xs.append(n)
        ys_raw.append(vals.mean())
        ys_corr.append((vals - ((m - 1) * 0.5 * np.log2(n) / n)).mean())

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_raw, marker="o", label=r"mean $-\frac{1}{n}\log_2 P(X_1^n)$")
    plt.plot(xs, ys_corr, marker="o", label=r"minus $(m-1)\frac{\log_2 n}{2n}$")
    plt.axhline(H_p, linestyle="--", label=r"$H(p)$")
    plt.xlabel(r"$n$")
    plt.ylabel("bits")
    plt.title("Finite-population: universal $(m-1)\\frac{\\log n}{2n}$ correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/finite_population_correction.pdf")
    plt.close()

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    ensure_figdir()
    fig_beta_bern_scatter(alpha=2.0, beta=5.0, n=400, trials=900, seed=0)
    fig_rigidity_prob_spread(p1=0.10, p2=0.45, w=0.5, ns=(60, 90, 120, 160, 220, 300), seed=1)
    fig_finite_population_correction(p=(0.2, 0.5, 0.3), N=4000,
                                     ns=(50, 80, 120, 180, 260, 360, 520),
                                     trials=250, seed=2)
