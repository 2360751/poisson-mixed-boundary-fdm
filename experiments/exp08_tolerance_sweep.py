"""Experiment 8 -- tolerance sweep: separating algebraic and discretisation error.

On one representative mesh (N=64, Neumann/ghost), solve with BiCGSTAB+ILU at
a sequence of relative-residual tolerances and record:
  * the discretisation error ||u_h - u*|| (should stop changing once the
    algebraic error drops below it),
  * the disagreement with the sparse-LU discrete oracle,
  * the scaled row residual actually achieved.

Then recompute the ghost-closure refinement table at two tolerances to show
the fitted order is a discretisation property, not a solver artefact.

Figures:
  fig06_tolerance_sweep.png -- error vs tolerance (log-log), plateau + oracle line
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIGURES, merge_results, fit_order

from fdm_mixed import assemble, solve_direct, error_norms, scaled_residual
from solvers_iterative import bicgstab_ilu


def sweep_tolerances(N=64, tols=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)):
    A, b, meta = assemble(N, closure="ghost", bc="neumann")
    u_ref, _ = solve_direct(A, b)
    e_ref = error_norms(u_ref, meta)
    rows = []
    for tol in tols:
        u, info = bicgstab_ilu(A, b, tol=tol)
        e = error_norms(u, meta)
        rows.append({
            "tol": tol,
            "achieved_rel_res": info["rel_res"],
            "iters": info["iters"],
            "disc_Linf": e["Linf"],
            "disc_L2": e["L2"],
            "err_vs_LU": float(np.max(np.abs(u - u_ref))),
            "scaled_resid": scaled_residual(A, b, u),
        })
    return rows, e_ref


def order_stability(ns=(16, 32, 64, 128), tol=1e-10):
    """Fitted order at a fixed tolerance: same as the direct-solver order."""
    l2, li = [], []
    for N in ns:
        A, b, meta = assemble(N, closure="ghost", bc="neumann")
        u, _ = bicgstab_ilu(A, b, tol=tol)
        e = error_norms(u, meta)
        l2.append(e["L2"])
        li.append(e["Linf"])
    hs = [1.0 / N for N in ns]
    return fit_order(hs, l2)[0], fit_order(hs, li)[0]


def main():
    rows, e_ref = sweep_tolerances()
    print(f"discretisation error (LU, N=64): Linf = {e_ref['Linf']:.3e}")
    for r in rows:
        print(f"tol={r['tol']:.0e}  achieved={r['achieved_rel_res']:.2e}  "
              f"iters={r['iters']:4d}  disc_Linf={r['disc_Linf']:.3e}  "
              f"vs_LU={r['err_vs_LU']:.3e}  scaled_resid={r['scaled_resid']:.2e}")

    # plateau check: disc error must stop changing for tol <= 1e-6
    early = rows[0]["disc_Linf"]
    late = rows[-1]["disc_Linf"]
    print(f"disc_Linf change tol 1e-2 -> 1e-12: {early:.3e} -> {late:.3e} "
          f"(ratio {early/late:.1f})")

    p2_a, pi_a = order_stability(tol=1e-6)
    p2_b, pi_b = order_stability(tol=1e-10)
    print(f"fitted ghost order at tol=1e-6:  L2 {p2_a:.3f}, Linf {pi_a:.3f}")
    print(f"fitted ghost order at tol=1e-10: L2 {p2_b:.3f}, Linf {pi_b:.3f}")

    tols = np.array([r["tol"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(tols, [r["disc_Linf"] for r in rows], "o-", label=r"discretisation $\|e\|_\infty$")
    ax[0].loglog(tols, [r["err_vs_LU"] for r in rows], "s--", label=r"algebraic $\|u-u_{LU}\|_\infty$")
    ax[0].loglog(tols, [r["scaled_resid"] for r in rows], "^:", label="scaled residual")
    ax[0].axhline(e_ref["Linf"], color="k", lw=0.8, ls="-.",
                  label="discretisation floor (LU)")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("solver tolerance (relative residual)")
    ax[0].set_ylabel("magnitude")
    ax[0].set_title("tolerance sweep, N=64, Neumann/ghost")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].semilogx(tols, [r["iters"] for r in rows], "o-")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("solver tolerance")
    ax[1].set_ylabel("BiCGSTAB+ILU iterations")
    ax[1].set_title("cost of tightening")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig06_tolerance_sweep.png"), dpi=300)
    plt.close(fig)

    merge_results({"exp08_tolerance_sweep": {
        "records": rows,
        "disc_floor_Linf": e_ref["Linf"],
        "order_tol_1e6": {"L2": p2_a, "Linf": pi_a},
        "order_tol_1e10": {"L2": p2_b, "Linf": pi_b},
    }})
    print("saved to figures/results.json")


if __name__ == "__main__":
    main()
