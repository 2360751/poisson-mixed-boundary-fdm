"""Experiment 1 -- hand-row verification and row-defect orders.

Pre-registered predictions (pre-computation protocol, written before running):
  P1. The one-sided boundary row defect is -(h/2) u*_xx(1,y) + O(h^2)
      = -(h/2) e^{1+y} + O(h^2):  fitted order 1 under refinement.
  P2. The ghost-eliminated PDE row defect is (h/3) u*_xxx(1,y) + O(h^2)
      = (h/3) e^{1+y} + O(h^2)  (Week-8 Taylor analysis): fitted order 1
      LOCALLY, although the global solution order stays 2.
  P3. The centred boundary condition (u_{N+1}-u_{N-1})/(2h) = q has defect
      (h^2/6) u*_xxx: fitted order 2.
  P4. Substituting u* into the interior five-point row gives -h^2/12
      (u_xxxx + u_yyyy) = -h^2 e^{x+y}/6: fitted order 2.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from common import fit_order, merge_results, FIGURES

from manufactured import u_star, f_source, BETA, q_neumann, r_robin


def interior_defect(h, x=0.5, y=0.5):
    """Substitute u* into the interior five-point row of -Laplace."""
    uP = u_star(x, y)
    uE = u_star(x + h, y)
    uW = u_star(x - h, y)
    uN = u_star(x, y + h)
    uS = u_star(x, y - h)
    lhs = (4 * uP - uE - uW - uN - uS) / h**2
    return lhs - f_source(x, y)


def onepsided_defect(h, y=0.5, bc="neumann"):
    data = q_neumann(y) if bc == "neumann" else r_robin(y)
    beta = BETA if bc == "robin" else 0.0
    lhs = (u_star(1.0, y) - u_star(1.0 - h, y)) / h + beta * u_star(1.0, y)
    return lhs - data


def ghost_defect(h, y=0.5, bc="neumann"):
    data = q_neumann(y) if bc == "neumann" else r_robin(y)
    beta = BETA if bc == "robin" else 0.0
    uN = u_star(1.0, y)
    uNm = u_star(1.0 - h, y)
    uS = u_star(1.0, y - h)
    uNb = u_star(1.0, y + h)
    lhs = (4 * uN - 2 * uNm - uS - uNb) / h**2 + 2 * beta * uN / h
    rhs = f_source(1.0, y) + 2 * data / h
    return lhs - rhs


def centred_bc_defect(h, y=0.5, bc="neumann"):
    data = q_neumann(y) if bc == "neumann" else r_robin(y)
    beta = BETA if bc == "robin" else 0.0
    centred = (u_star(1.0 + h, y) - u_star(1.0 - h, y)) / (2 * h)
    return centred - (data - beta * u_star(1.0, y))


def main():
    Ns = [16, 32, 64, 128, 256, 512]
    hs = [1.0 / n for n in Ns]
    rows = {}
    for name, fn, extra in [
        ("interior", interior_defect, {}),
        ("onesided_neumann", onepsided_defect, {"bc": "neumann"}),
        ("ghost_neumann", ghost_defect, {"bc": "neumann"}),
        ("centred_bc_neumann", centred_bc_defect, {"bc": "neumann"}),
        ("onesided_robin", onepsided_defect, {"bc": "robin"}),
        ("ghost_robin", ghost_defect, {"bc": "robin"}),
    ]:
        vals = [abs(fn(h, **extra)) for h in hs]
        order, dev = fit_order(hs, vals)
        rows[name] = {"abs_defect": vals, "fitted_order": order,
                      "fit_dev": dev}
        print(f"{name:18s} order={order:5.3f}  " +
              " ".join(f"{v:.3e}" for v in vals))

    # predicted constants at y = 0.5:  u*_xx = u*_xxx = u*_xxxx = e^{1.5}
    e15 = np.exp(1.5)
    print(f"\npredicted constants at y=0.5 (e^1.5 = {e15:.4f}):")
    print(f"  onesided defect / h  -> -e/2 = {-e15/2:.4f}   "
          f"measured {rows['onesided_neumann']['abs_defect'][2]/hs[2] if False else onepsided_defect(hs[2])/hs[2]:.4f}")
    print(f"  ghost defect / h     ->  e/3 = {e15/3:.4f}   "
          f"measured {ghost_defect(hs[2])/hs[2]:.4f}")
    print(f"  centred defect / h^2 ->  e/6 = {e15/6:.4f}   "
          f"measured {centred_bc_defect(hs[2])/hs[2]**2:.4f}")
    print(f"  interior defect / h^2-> -e/6 = {-e15/6:.4f}   "
          f"measured {interior_defect(hs[2])/hs[2]**2:.4f}")

    merge_results({"exp01_row_defects": rows})
    print("\nsaved to figures/results.json")


if __name__ == "__main__":
    main()
