r"""Estimation of zero-inflated binomial distribution.

Code for the paper "Parametric Models of Linguistic Count Data" from
ACL 2003 (http://www.aclweb.org/anthology/P03-1037).

Note that the pseudocode in Figure 5 of that paper contains a mistake
on line 5: In place of the published line

  \hat{z}_i \gets z / (z + (1-p)^{n_i})

it should read

  \hat{z}_i \gets z / (z + (1-z)(1-p)^{n_i})

"""

from __future__ import division

import math
import random
import sys

def em_iteration(z, p, data):
    """One EM iteration for estimating zero-inflated binomial parameters."""
    s = len(data)
    Z = 0
    X = 0
    N = 0
    L = 0
    # E step
    for xi, ni in data:
        if xi == 0:
            dbinom_0 = (1 - p)**ni
            Pr_xi = z + (1 - z) * dbinom_0
            L += log(Pr_xi)
            zi = z / Pr_xi
        else:
            L += log(1 - z) + xi * log(p) + (ni - xi) * log(1 - p)
            zi = 0
        Z += zi
        X += (1 - zi) * xi
        N += (1 - zi) * ni
    # M step
    z = Z / s
    p = X / N
    return z, p, L

INF = 1e300 * 1e300

def log(x):
    """Logarithm function that doesn't raise an exception for log(0)."""
    if x == 0:
        return -INF
    return math.log(x)

def rbernoulli(rng, p):
    """Bernoulli variate."""
    return rng.uniform(0, 1) < p

def rbinom(rng, n, p):
    """Binomial variate."""
    k = 0
    for _ in range(n):
        if rbernoulli(rng, p):
            k += 1
    return k

def rzib(rng, n, z, p):
    """Zero-inflated binomial variate."""
    if rbernoulli(rng, z):
        return 0
    return rbinom(rng, n, p)

def main(argv):
    """Estimate ZIB distribution on data sampled from true distribution."""
    true_z = float(argv[1])
    true_p = float(argv[2])

    # Artificial data sampled from true distribution.
    data = []
    rng = random.Random()
    for _ in range(10000):
        n = rng.randint(1, 12)
        k = rzib(rng, n, true_z, true_p)
        data.append((k, n))

    # Estimate parameters based on artificial data.
    # Initial guess: all zeroes come from ZI component,
    # all non-zero values come from binomial component.
    z = 1
    p = 0.5
    prev_L = -INF
    for _ in range(1000):
        z, p, L = em_iteration(z, p, data)
        sys.stdout.write('z = %f  p = %f  L = %.6f\n' % (z, p, L))
        assert 0 <= z <= 1
        assert 0 <= p <= 1
        assert L >= prev_L
        if L - prev_L < 1e-7:
            break
        prev_L = L
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
