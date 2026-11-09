"""Iterative solvers for the five-point mixed-BC system (skeleton completed).

Stopping criterion: RELATIVE RESIDUAL ||b - A x||_inf / ||b - A x0||_inf,
never the iterate difference (the Week-6 failure demo showed an iterate
difference can stop far from the solution).

Note on CG: the mixed-BC assembly is NONSYMMETRIC (first-order boundary rows
have h^-1 scaling; ghost elimination doubles the inward coupling).  We
therefore provide Jacobi / Gauss-Seidel / SOR, and a nonsymmetric Krylov
method (BiCGSTAB + ILU) as the library reference.  Conjugate gradients is
deliberately absent -- see the report's solver discussion.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def run_sweep(A, b, sweep, tol=1e-8, max_iter=20000, x0=None):
    """Run `sweep(x) -> x_new` until rel. residual < tol.  Returns info dict."""
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    r0 = np.linalg.norm(b - A @ x, np.inf)
    if r0 == 0.0:
        return x, {"iters": 0, "rel_res": 0.0, "history": [0.0], "time": 0.0,
                   "converged": True}
    t0 = time.perf_counter()
    history = [1.0]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        x = sweep(x)
        rel = np.linalg.norm(b - A @ x, np.inf) / r0
        history.append(rel)
        if rel < tol:
            converged = True
            break
    dt = time.perf_counter() - t0
    return x, {"iters": it, "rel_res": history[-1], "history": history,
               "time": dt, "converged": converged}


def jacobi(A, b):
    """x^{k+1} = D^{-1} (b - (A - D) x^k), vectorised via the residual."""
    d = A.diagonal()
    if np.any(d == 0):
        raise ValueError("zero diagonal in Jacobi")

    def sweep(x):
        r = b - A @ x
        return x + r / d

    return sweep


def gauss_seidel(A, b):
    """One sweep solves L x_new = b - U x with L = tril(A), U = triu(A,1)."""
    L = sp.tril(A).tocsr()
    U = sp.triu(A, k=1).tocsr()
    lu = spla.splu(L.tocsc())

    def sweep(x):
        return lu.solve(b - U @ x)

    return sweep


def sor(A, b, omega=1.5):
    """x^{k+1} = x^k + omega D^{-1} (b - A x^k) + (1-omega) update via GS split.

    Implemented as damped Gauss-Seidel: solve (D/omega + L) x_new =
    b - U x + (1-omega)/omega D x, which is the standard SOR sweep.
    """
    D = sp.diags(A.diagonal())
    L = sp.tril(A, k=-1).tocsr()
    U = sp.triu(A, k=1).tocsr()
    M = (D / omega + L).tocsc()
    lu = spla.splu(M)

    def sweep(x):
        rhs = b - U @ x + ((1.0 - omega) / omega) * A.diagonal() * x
        return lu.solve(rhs)

    return sweep


def bicgstab_ilu(A, b, tol=1e-8, max_iter=2000):
    """Nonsymmetric Krylov reference with ILU preconditioner."""
    A = A.tocsc()
    ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=10)
    M = spla.LinearOperator(A.shape, ilu.solve)
    x0 = np.zeros(b.shape[0])
    r0 = np.linalg.norm(b - A @ x0, np.inf)
    counter = {"iters": 0}

    def cb(xk):
        counter["iters"] += 1

    t0 = time.perf_counter()
    x, info = spla.bicgstab(A, b, x0=x0, rtol=tol, atol=0.0,
                            maxiter=max_iter, M=M, callback=cb)
    dt = time.perf_counter() - t0
    rel = np.linalg.norm(b - A @ x, np.inf) / r0 if r0 > 0 else 0.0
    return x, {"iters": counter["iters"], "rel_res": rel, "time": dt,
               "converged": info == 0, "method": "bicgstab+ILU"}


def match_tol_to_discretisation(h, c=1.0e-4):
    """Working algebraic tolerance scaled with the discretisation scale.

    Week-7 lesson: a FIXED tolerance lets the algebraic error grow relative
    to the discretisation error like h^-2; tol ~ C h^2 keeps alg/disc stable.
    Here we return a relative-residual target ~ C h^2.
    """
    return c * h * h
