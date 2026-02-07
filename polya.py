import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def product(n, function):
    """
    Computes the partial product from k=0 to k=N of FUNCTION(k).
    >>> product(3, math.exp2)
    64.0
    """
    total, k = 1, 0
    while k <= n:
        total, k = total * function(k), k + 1
    return total

def joint_polya_prob(b, w, a, sequence):
    """
    Given a Polya's urn problem with B black balls, W white balls, A + 1 replacement balls after each draw,
    computes the joint probability of the SEQUENCE of 1s and 0s where 1 = black ball, 0 = white ball.
    >>> joint_polya_prob(10, 10, 1, [0])
    0.5
    >>> joint_polya_prob(10, 10, 1, [0,1,1])
    0.11904761904761904
    >>> joint_polya_prob(10, 10, 1, [1,1,0])
    0.11904761904761904
    >>> joint_polya_prob(10, 20, 1, [1,1,0])
    0.0739247311827957
    """
    n = len(sequence)
    k = sum(sequence)
    def num1function(x):
        return b + a * x
    def num2function(x):
        return w + a * x
    def denfunction(x):
        return b + w + a * x
    num = product(k-1, num1function) * product(n-k-1, num2function)
    den = product(n-1, denfunction)
    return num / den

def entropy(b, w):
    """
    Given a Polya's urn problem with B black balls, W white balls, calculates the entropy of one draw
    (equivalent to weighted coinflip with P(Heads) = b/(b+w)).
    >>> entropy(1, 1)
    1.0
    >>> entropy(10, 5)
    0.9182958340544896
    >>> entropy(100, 1)
    0.0801360473312753
    """
    p = b/(b+w)
    return - (p * math.log(p, 2) + (1-p) * math.log(1 - p, 2))

def eps_typical(b, w, a, sequence, eps):
    """
    Given a Polya's urn problem with B black balls, W white balls, A + 1 replacement balls after each draw,
    determines whether the SEQUENCE of 1s and 0s where 1 = black ball, 0 = white ball is EPS-typical.
    >>> eps_typical(10, 10, 1, [0], 0.1)
    True
    >>> eps_typical(10, 12, 1, [0,1,1,0,1], 0.1)
    True
    """
    n = len(sequence)
    H = entropy(b,w)
    left_bound = 2 ** (-n * (H + eps) )
    right_bound = 2 ** (-n * (H - eps) )
    if left_bound <= joint_polya_prob(b, w, a, sequence) <= right_bound:
        return True
    return False

def prob_eps_typical(b, w, a, n, eps):
    """
    Given a Polya's urn problem with B black balls, W white balls, A + 1 replacement balls after each draw,
    calculates the probability that a random sequence of length N is EPS-typical.
    Due to exchangeability, instead of testing all sequences, we simply have to test cases where the first k
    balls are black and the last N - k balls are white.
    >>> prob_eps_typical(10, 10, 1, 2, 0.1)
    1.0
    >>> prob_eps_typical(10, 20, 1, 100, 0.1)
    0.7223628048235586
    >>> prob_eps_typical(10, 12, 5, 100, 0.1)
    0.6083467375875602
    """
    k, prob = 0, 0
    while k <= n:
        sequence = [1]*k + [0]*(n-k)
        if eps_typical(b, w, a, sequence, eps):
            n_choose_k = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
            prob = prob + n_choose_k * joint_polya_prob(b, w, a, sequence)
        k = k + 1
    return prob


def graph_eps_typical(b, w, a, n_max, eps):
    """
    Given a Polya's urn problem with B black balls, W white balls, A + 1 replacement balls after each draw,
    calculates the probability that a random sequence of length n is EPS-typical for n from 1 to N_MAX.
    Creates a scatter plot of these probabilities over n, with a bar at y = 1 - EPS.
    """
    n = np.array( list( range(1, n_max + 1) ) )
    i, f_n = 1, []
    while i <= n_max:
        f_n.append(prob_eps_typical(b, w, a, i, eps))
        i = i+1
    f_n = np.array(f_n)
    plt.scatter(n, f_n)
    plt.title("Probability a sequence of length n is epsilon-typical vs. n")
    plt.xlabel("n")
    plt.ylabel("Probability a sequence of length n is epsilon-typical")
    plt.plot([1, n_max], [1-eps, 1-eps], 'k-', lw=2)
    plt.show()
    return ""


"""EXTRA:"""

def polya(b, w, a, n):
    """
    Given an urn with B black balls and W white balls, draws a ball u.a.r.
    and prints the color x. Adds A + 1 balls of color x back to the urn. Repeats N times.
    >>> polya(10, 10, 1, 5)
    white 
    black
    black
    black
    white 
    >>> polya(10, 10, 1, 5)
    white 
    black
    black
    white
    white
    """
    k, B, W = 1, b, w
    while k <= n:
        prob = B / (B+W)
        if random.random() < prob:
            current_ball = "black"
            B = B + a
        else:
            current_ball = "white "
            W = W + a
        print(current_ball)
        k = k+1


def polya_simulate(b, w, a, n, rng=None):
    """
    Simulates a Pólya urn with B black balls, W white balls, and A+1 replacement balls.
    Returns (sequence, B_history, W_history) where:
    - sequence: list of 1s (black) and 0s (white)
    - B_history, W_history: counts after each draw (includes initial state)
    """
    if rng is None:
        rng = random
    sequence = []
    B_history = [b]
    W_history = [w]
    B, W = b, w
    for _ in range(n):
        prob = B / (B + W)
        if rng.random() < prob:
            sequence.append(1)
            B += a
        else:
            sequence.append(0)
            W += a
        B_history.append(B)
        W_history.append(W)
    return sequence, B_history, W_history


def visualize_polya(b, w, a, n, num_runs=5, seed=None, save_path=None):
    """
    Visualizes Pólya urn simulations.
    - Left: proportion of black balls in urn over time (with initial ratio as reference)
    - Right: empirical proportion of black draws (cumulative mean) over time
    Runs multiple simulations to show path-dependence and reinforcement.
    """
    if seed is not None:
        random.seed(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    initial_ratio = b / (b + w)
    draws = np.arange(1, n + 1)
    for run in range(num_runs):
        seq, B_hist, W_hist = polya_simulate(b, w, a, n)
        B_arr = np.array(B_hist[1:])  # composition after each draw
        W_arr = np.array(W_hist[1:])
        prop_black_urn = B_arr / (B_arr + W_arr)
        cumsum_black = np.cumsum(seq)
        prop_black_drawn = cumsum_black / draws
        axes[0].plot(draws, prop_black_urn, alpha=0.7)
        axes[1].plot(draws, prop_black_drawn, alpha=0.7)
    axes[0].axhline(initial_ratio, color="gray", linestyle="--", label=f"Initial = {initial_ratio:.2f}")
    axes[0].set_xlabel("Draw")
    axes[0].set_ylabel("Proportion of black balls in urn")
    axes[0].set_title("Urn composition over time")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(initial_ratio, color="gray", linestyle="--", label=f"Initial = {initial_ratio:.2f}")
    axes[1].set_xlabel("Draw")
    axes[1].set_ylabel("Cumulative proportion of black draws")
    axes[1].set_title("Empirical proportion of black draws")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Pólya urn: B={b}, W={w}, a={a}, n={n} draws ({num_runs} runs)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig


if __name__ == "__main__":
    graph_eps_typical(b=10, w = 10, a=10, n_max = 200, eps = 0.2)
