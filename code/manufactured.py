"""Manufactured data for Project 2, Direction 2 (mixed boundary data).

Problem (the reproducible model from the Problem Pack, Direction 2):

    -Laplace u = f  in Omega = (0,1)^2
                 u = g   on Gamma_D = {x=0} U {y=0} U {y=1}  (all four corners)
    Neumann :  d_n u = q  on Gamma_B = {1} x (0,1)
    Robin   :  d_n u + beta*u = r  on Gamma_B,  beta = 2

    u*(x,y) = exp(x+y),  f = -2 exp(x+y),
    g = u*|_GammaD,  q(y) = exp(1+y),  r(y) = 3 exp(1+y).

Sign convention (stated once, used everywhere):
    the operator is -Laplace (positive diagonal after discretisation),
    d_n is the OUTWARD normal derivative, so on x=1  d_n u = u_x(1,y).
"""
import numpy as np

BETA = 2.0


def u_star(x, y):
    return np.exp(x + y)


def f_source(x, y):
    """f = -Delta u* = -2 exp(x+y)."""
    return -2.0 * np.exp(x + y)


def g_dirichlet(x, y):
    """Dirichlet data on x=0, y=0, y=1 (including all four corners)."""
    return np.exp(x + y)


def q_neumann(y):
    """Outward normal derivative on x=1: u_x(1,y) = exp(1+y) > 0."""
    return np.exp(1.0 + y)


def r_robin(y):
    """Robin data on x=1: d_n u + beta u = (1+beta) exp(1+y), beta = 2."""
    return (1.0 + BETA) * np.exp(1.0 + y)


# --- pure-Neumann depth variant: d_n u = qN on the WHOLE boundary ------
# n = (-1,0) on x=0, (0,-1) on y=0, (0,+1) on y=1, (+1,0) on x=1.

def q_pure_west(y):
    """d_n u on x=0, n=(-1,0): -u_x(0,y) = -exp(y)."""
    return -np.exp(y)


def q_pure_south(x):
    """d_n u on y=0, n=(0,-1): -u_y(x,0) = -exp(x)."""
    return -np.exp(x)


def q_pure_north(x):
    """d_n u on y=1, n=(0,+1): u_y(x,1) = exp(x+1)."""
    return np.exp(x + 1.0)


def q_pure_east(y):
    return q_neumann(y)


def check_forcing_by_fd(n=200, tol=5e-3):
    """Self-check: verify f = -Delta u* by finite differences on a fine grid.

    Independent of any project assembly code -- the point is to catch typos
    in the analytic data (the 'analytic bank' gate of the brief).
    """
    h = 1.0 / n
    x = np.linspace(h, 1.0 - h, n - 1)  # interior only, avoids boundary bias
    X, Y = np.meshgrid(x, x, indexing="ij")
    U = u_star(X, Y)
    lap = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / h**2 \
        + (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / h**2
    f_fd = -lap
    f_ex = f_source(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    scale = np.abs(f_ex).max()
    rel = np.abs(f_fd - f_ex).max() / scale
    assert rel < tol, f"forcing self-check failed: rel={rel:.2e}"
    return rel


def check_robin_data(tol=1e-12):
    """r = q + beta*u on x=1 must hold identically."""
    y = np.linspace(0.0, 1.0, 17)
    resid = np.abs(r_robin(y) - (q_neumann(y) + BETA * u_star(1.0, y))).max()
    assert resid < tol, f"robin data inconsistent: {resid:.2e}"
    return resid


if __name__ == "__main__":
    print(f"forcing FD self-check rel err = {check_forcing_by_fd():.3e}")
    print(f"robin data consistency        = {check_robin_data():.3e}")
