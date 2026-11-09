"""Smoke test: assembly sanity + expected orders on a small mesh sequence.

Run:  python tests/test_smoke.py
Expectation (pre-registered before running -- pre-computation protocol):
  * manufactured data self-checks pass;
  * hand-row defects decay: onesided O(h), ghost O(h) locally (boundary row)
    but GLOBAL solution order: onesided -> ~1, ghost -> ~2 (Week-8 result);
  * pure-Neumann: A @ 1 = 0 exactly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from manufactured import check_forcing_by_fd, check_robin_data
from fdm_mixed import (assemble, solve_direct, error_norms, scaled_residual,
                       row_defects, to_grid)
import fdm_mixed as fm


def fit_order(hs, errs):
    hs = np.asarray(hs, float)
    errs = np.asarray(errs, float)
    p = np.polyfit(np.log(hs), np.log(errs), 1)
    return p[0]


def main():
    print("== analytic data self-checks ==")
    print(f"  forcing FD rel err   : {check_forcing_by_fd():.3e}")
    print(f"  robin data residual  : {check_robin_data():.3e}")

    print("\n== pure-Neumann null vector ==")
    A, b, meta = assemble(8, closure="ghost", pure_neumann=True)
    ones = np.ones(meta["n"])
    print(f"  ||A @ 1||_inf = {np.abs(A @ ones).max():.3e}  (expect 0)")

    print("\n== hand-row defects (N=64, y=0.5) ==")
    for bc in ("neumann", "robin"):
        for cl in ("onesided", "ghost"):
            d = row_defects(64, cl, bc)
            print(f"  {bc:7s} {cl:9s}: {d}")

    print("\n== global convergence (ghost, neumann) -- expect ~2 ==")
    for N in (16, 32, 64, 128):
        A, b, meta = assemble(N, closure="ghost", bc="neumann")
        u, info = solve_direct(A, b)
        e = error_norms(u, meta)
        sr = scaled_residual(A, b, u)
        print(f"  N={N:4d}  L2={e['L2']:.4e}  Linf={e['Linf']:.4e}  "
              f"bnd={e['bnd_Linf']:.4e}  scaled_res={sr:.2e}")

    print("\n== global convergence (onesided, neumann) -- expect ~1 ==")
    errs, hs = [], []
    for N in (16, 32, 64, 128):
        A, b, meta = assemble(N, closure="onesided", bc="neumann")
        u, info = solve_direct(A, b)
        e = error_norms(u, meta)
        errs.append(e["Linf"])
        hs.append(meta["h"])
        print(f"  N={N:4d}  L2={e['L2']:.4e}  Linf={e['Linf']:.4e}  "
              f"bnd={e['bnd_Linf']:.4e}")
    print(f"  fitted Linf order (onesided) = {fit_order(hs, errs):.3f}")

    print("\n== robin ghost -- expect ~2 ==")
    errs, hs = [], []
    for N in (16, 32, 64, 128):
        A, b, meta = assemble(N, closure="ghost", bc="robin")
        u, info = solve_direct(A, b)
        e = error_norms(u, meta)
        errs.append(e["L2"])
        hs.append(meta["h"])
        print(f"  N={N:4d}  L2={e['L2']:.4e}  Linf={e['Linf']:.4e}  "
              f"bnd={e['bnd_Linf']:.4e}")
    print(f"  fitted L2 order (robin ghost) = {fit_order(hs, errs):.3f}")


if __name__ == "__main__":
    main()
