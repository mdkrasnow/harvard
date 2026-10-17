"""Reproduce the numerical calculations and plot data in PSet2.tex.

Run from this directory with: python3 pset2_analysis.py
Requires NumPy. The two CSV files are read by pgfplots in the LaTeX document.
"""

import csv
import numpy as np


def double_descent():
    rng = np.random.default_rng(286)
    x = np.linspace(0.05, 0.95, 10) + rng.uniform(-0.02, 0.02, 10)
    y = rng.normal(size=10)
    # 1,000 fresh sets of 10 points; their noise is independent of training.
    test_x = rng.uniform(0, 1, 10000)
    test_y = rng.normal(size=10000)
    ns = [1, 2, 3, 5, 7, 9, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    lambdas = [0.0, 0.0001, 0.01, 1.0]
    rows = []
    for n in ns:
        k = np.arange(n)
        train = np.exp(1j * x[:, None] * k)
        test = np.exp(1j * test_x[:, None] * k)
        row = [n]
        for lam in lambdas:
            if lam == 0:
                coef = np.linalg.lstsq(train, y, rcond=1e-12)[0]
            else:
                gram = train @ train.conj().T
                coef = train.conj().T @ np.linalg.solve(
                    gram + lam * np.eye(10), y
                )
            row.append(np.mean(np.abs(test @ coef - test_y) ** 2))
        rows.append(row)
    with open("pset2_double_descent.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "zero", "small", "medium", "large"])
        writer.writerows(rows)
    print("Double descent (N, lambda=0, 1e-4, 0.01, 1):")
    for row in rows:
        if row[0] in (1, 10, 50, 500):
            print(row)


def entropy():
    widths = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    columns = []
    for name in ("points", "noisypoints"):
        data = np.loadtxt(name + ".txt")
        values = []
        for width in widths:
            cells = np.floor(data / width).astype(np.int32)
            _, counts = np.unique(cells, axis=0, return_counts=True)
            p = counts / len(data)
            values.append(-np.sum(p * np.log2(p)))
        columns.append(values)
    with open("pset2_entropy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "clean", "noisy"])
        writer.writerows(zip(widths, *columns))
    print("Entropy (width, clean, noisy):")
    for row in zip(widths, *columns):
        print(row)


def coin():
    flips = np.loadtxt("coinflips.txt")
    grid = np.linspace(0, 1, 100001)
    for n in (10, 100, 1000):
        h = int(flips[:n].sum())
        # Endpoints are excluded for the observed mixed-head/tail samples.
        p = grid[1:-1]
        log_likelihood = h * np.log(p) + (n - h) * np.log1p(-p)
        gaussian = log_likelihood - (p - 0.5) ** 2 / 0.08
        bimodal_prior = 1 / (1 + 25 * p**2) + 1 / (1 + 25 * (1 - p) ** 2)
        bimodal = log_likelihood + np.log(bimodal_prior)
        print("Coin:", n, h, h / n, p[np.argmax(gaussian)], p[np.argmax(bimodal)])


def fluorescence():
    rng = np.random.default_rng(198)
    for suffix in ("2a", "2b", "2c1", "2c2"):
        data = np.loadtxt("fluordata_" + suffix + ".txt", skiprows=1)

        def estimate(sample):
            mother, first, second = sample.T
            total = first + second
            difference = first - second
            survival = np.sum(mother * total) / np.sum(mother**2)
            imbalance = np.sum(difference) / np.sum(total)
            adjusted = (difference - imbalance * total) ** 2
            if suffix == "2b":
                brightness = np.mean(adjusted / total) / (1 - imbalance**2)
                noise_intercept = 0.0
            elif suffix == "2c2":
                # D^2 = f*S + 2*sigma^2: the intercept removes readout noise.
                noise_intercept, brightness = np.linalg.lstsq(
                    np.column_stack((np.ones(len(total)), total)),
                    adjusted,
                    rcond=None,
                )[0]
            else:
                brightness = np.mean(adjusted / total)
                noise_intercept = 0.0
            return np.array((survival, imbalance, brightness, noise_intercept))

        fit = estimate(data)
        boot = np.array([estimate(data[rng.integers(len(data), size=len(data))])
                         for _ in range(1000)])
        se = boot.std(axis=0, ddof=1)
        print("Fluorescence:", suffix, "estimate", fit, "bootstrap SE", se)


if __name__ == "__main__":
    double_descent()
    entropy()
    coin()
    fluorescence()
