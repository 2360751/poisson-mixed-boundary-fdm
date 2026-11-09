"""Experiment 6 -- pure-Neumann depth (brief tier ⋆⋆⋆).

Demonstrates, on the whole-boundary Neumann variant of u* = e^{x+y}:
  N1. the constant vector spans the right null space of A (A @ 1 = 0 exactly);
  N2. the left null vector w (null of A^T) is computed on a small mesh and
      the discrete compatibility defect |<b, w>| / (||b|| ||w||) is reported;
  N3. with a pinned gauge the solve succeeds; after removing the arbitrary
      constant the solution matches u* at the expected rate (O(h) global with
      the ghost rows whose local defect is O(h), measured not assumed);
  N4. falsification: with an INCOMPATIBLE rhs (add +1 to f) the pinned solve
      returns a solution with a large defect in the pinned row -- the
      discrete compatibility condition is what makes the pure-Neumann
      problem solvable.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIGURES, merge_results, fit_order

from fdm_mixed import (assemble, solve_pinned_direct, to_grid, exact_grid,
                       raw_residual_inf)
from manufactured import u_star


def best_shift(U, meta):
    E0 = U - exact_grid(meta)
    return -E0.mean()


def run(N=32):
    A, b, meta = assemble(N, closure="ghost", pure_neumann=True)
    n = meta["n"]
    ones = np.ones(n)
    null_res = float(np.abs(A @ ones).max())

    # left null vector via dense SVD on a coarse system only
    left_defect = None
    w_spectrum = None
    if n <= 2500:
        Ad = A.toarray()
        uu, ss, vv = np.linalg.svd(Ad)
        w = uu[:, -1]
        left_defect = float(np.abs(A.T @ w).max())
        w_spectrum = ss[-3:].tolist()
        compat = float(np.abs(b @ w) / (np.linalg.norm(b) * np.linalg.norm(w)))
    else:
        compat = None

    # pinned solve
    u, info = solve_pinned_direct(A, b, meta, pin_k=0)
    resid_orig = raw_residual_inf(A, b, u)   # defect in the pinned row
    U = to_grid(u, meta)
    shift = best_shift(U, meta)
    E = (U + shift) - exact_grid(meta)
    err_shifted = {"Linf": float(np.abs(E).max()),
                   "L2": float(meta["h"] * np.linalg.norm(E.ravel(), 2))}
    err_raw = float(np.abs(U - exact_grid(meta)).max())

    # falsification: incompatible rhs
    b_bad = b + 1.0
    u_bad, _ = solve_pinned_direct(A, b_bad, meta, pin_k=0)
    resid_bad = raw_residual_inf(A, b_bad, u_bad)

    print(f"N={N}  n_unknown={n}")
    print(f"  ||A @ 1||_inf               = {null_res:.3e}   (expect 0)")
    if left_defect is not None:
        print(f"  ||A^T w||_inf (w=left null)  = {left_defect:.3e}")
        print(f"  smallest 3 singular values  = {w_spectrum}")
        print(f"  compatibility |<b,w>|/(|b||w|) = {compat:.3e}  (expect ~0)")
    print(f"  pinned-row defect ||Au-b||_inf = {resid_orig:.3e}")
    print(f"  raw Linf (before shift)        = {err_raw:.4e}")
    print(f"  after mean-shift: Linf={err_shifted['Linf']:.4e}  "
          f"L2={err_shifted['L2']:.4e}")
    print(f"  FALSIFICATION incompatible b: pinned-row defect = {resid_bad:.3e} "
          f"(vs {resid_orig:.3e})")

    return {"N": N, "n": n, "null_res": null_res, "left_defect": left_defect,
            "compatibility": compat, "pinned_defect": resid_orig,
            "err_shifted": err_shifted, "err_raw": err_raw,
            "falsification_defect": resid_bad}


def convergence(ns=(16, 32, 64, 128, 256)):
    recs = []
    for N in ns:
        A, b, meta = assemble(N, closure="ghost", pure_neumann=True)
        u, _ = solve_pinned_direct(A, b, meta, pin_k=0)
        U = to_grid(u, meta)
        shift = best_shift(U, meta)
        E = (U + shift) - exact_grid(meta)
        recs.append({"N": N, "h": meta["h"],
                     "L2": float(meta["h"] * np.linalg.norm(E.ravel(), 2)),
                     "Linf": float(np.abs(E).max())})
        print(f"  N={N:4d}  L2={recs[-1]['L2']:.4e}  Linf={recs[-1]['Linf']:.4e}")
    p2, _ = fit_order([r["h"] for r in recs], [r["L2"] for r in recs])
    pi, _ = fit_order([r["h"] for r in recs], [r["Linf"] for r in recs])
    print(f"  fitted: L2 order {p2:.3f}, Linf order {pi:.3f}")
    return recs, p2, pi


def main():
    print("=== null space & compatibility, N=32 ===")
    small = run(32)
    print("\n=== convergence with pinned gauge + mean shift ===")
    recs, p2, pi = convergence()
    merge_results({"exp06_pure_neumann": {"small": small, "convergence": recs,
                                          "fit_L2": p2, "fit_Linf": pi}})
    print("saved to figures/results.json")


if __name__ == "__main__":
    main()
