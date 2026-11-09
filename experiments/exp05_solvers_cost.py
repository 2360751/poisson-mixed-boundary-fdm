"""Experiment 5 -- direct vs iterative at matched accuracy; cost vs accuracy.

Cost measure (named before quoting): (a) assembly wall time, (b) solve wall
time, (c) nnz / factorisation fill, (d) iteration count for iterative runs.
Deliberately NOT counted: interpreter start-up, figure rendering, the
one-off splu object construction outside the timed solve.

Accuracy target (frozen across configurations): the DISCRETE solution of the
same linear system -- i.e. all solvers must reach relative residual <= 1e-10
on the SAME assembled system, so their solutions agree to ~cond(A)*1e-10;
we additionally verify ||u_iter - u_splu||_inf directly.

Predictions:
  S1. splu is fastest at these sizes (n <= ~6.6e4) but fills in more memory;
      GS/SOR need O(N^2) sweeps to reduce the residual by 4 orders (spectral
      radius 1 - O(h^2)); Jacobi is slower still.
  S2. At matched algebraic accuracy the DISCRETISATION error does not change
      -- only the cost does.  Repeating the direct solve with an extra-tight
      ILU cross-check changes the solution by << discretisation error.
  S3. Cost-accuracy for closures: reaching ||e||_inf <= 1e-4 on the right
      boundary costs N~123 (ghost, extrapolated) vs N~3743 (onesided) as in
      Week 8; we measure the actual work up to N=1024 and extrapolate.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIGURES, merge_results, fit_order

from fdm_mixed import assemble, solve_direct, error_norms, scaled_residual
import solvers_iterative as si


def solver_race(N=64, bc="neumann", closure="ghost", tol=1e-10):
    A, b, meta = assemble(N, closure=closure, bc=bc)
    t0 = time.perf_counter()
    u_ref, info_ref = solve_direct(A, b)
    t_direct = time.perf_counter() - t0
    rows = [{"solver": "splu", "iters": 1, "time": t_direct,
             "rel_res": info_ref["resid_inf"] / np.abs(b).max(),
             "err_vs_ref": 0.0}]
    for name, mk in [("jacobi", si.jacobi), ("gauss_seidel", si.gauss_seidel),
                     ("sor(1.5)", lambda A, b: si.sor(A, b, 1.5))]:
        sweep = mk(A, b)
        x, inf = si.run_sweep(A, b, sweep, tol=tol, max_iter=200000)
        rows.append({"solver": name, "iters": inf["iters"], "time": inf["time"],
                     "rel_res": inf["rel_res"],
                     "err_vs_ref": float(np.abs(x - u_ref).max()),
                     "converged": inf["converged"]})
    x, inf = si.bicgstab_ilu(A, b, tol=tol)
    rows.append({"solver": "bicgstab+ILU", "iters": inf["iters"],
                 "time": inf["time"], "rel_res": inf["rel_res"],
                 "err_vs_ref": float(np.abs(x - u_ref).max()),
                 "converged": inf["converged"]})
    for r in rows:
        print(f"    {r['solver']:14s} iters={r['iters']:7d}  "
              f"time={r['time']:.3f}s  rel_res={r['rel_res']:.2e}  "
              f"|du|_inf={r['err_vs_ref']:.2e}")
    return rows


def closure_cost_accuracy(bc="neumann", ns=(16, 32, 64, 128, 256, 512)):
    """Work each closure must spend to reach boundary Linf targets.

    N=512 is the largest direct run (n = 262k unknowns); N_needed beyond
    that is EXTRAPOLATED from the fitted order, as in the Week-8 study.
    """
    targets = [1e-2, 1e-3, 1e-4, 1e-5]
    out = {}
    for closure in ("ghost", "onesided"):
        recs = []
        for N in ns:
            t0 = time.perf_counter()
            A, b, meta = assemble(N, closure=closure, bc=bc)
            t_asm = time.perf_counter() - t0
            t0 = time.perf_counter()
            u, info = solve_direct(A, b)
            t_sol = time.perf_counter() - t0
            e = error_norms(u, meta)
            recs.append({"N": N, "h": meta["h"], "bnd_Linf": e["bnd_Linf"],
                         "Linf": e["Linf"], "L2": e["L2"], "nnz": A.nnz,
                         "t_assemble": t_asm, "t_solve": t_sol})
            print(f"    {closure:9s} N={N:5d}  bnd_Linf={e['bnd_Linf']:.4e}  "
                  f"t_solve={t_sol:.3f}s  nnz={A.nnz}")
        # extrapolate N needed for each target via fitted order on last 3
        p, _ = fit_order([r["h"] for r in recs[-3:]],
                         [r["bnd_Linf"] for r in recs[-3:]])
        last = recs[-1]
        need = {}
        for tgt in targets:
            # e ~ C h^p  ->  N_need = N_last * (e_last/tgt)^(1/p)
            n_need = last["N"] * (last["bnd_Linf"] / tgt) ** (1.0 / p)
            need[f"{tgt:.0e}"] = float(n_need)
        out[closure] = {"records": recs, "fitted_order": p,
                        "N_needed_by_target": need}
        print(f"    -> fitted boundary order {p:.3f}, N_needed: {need}")
    return out


def main():
    print("=== solver race, N=64, Neumann ghost, tol=1e-10 ===")
    race = solver_race(64, "neumann", "ghost", 1e-10)
    print("=== solver race, N=64, Neumann onesided (h^-1 rows stress) ===")
    race2 = solver_race(64, "neumann", "onesided", 1e-10)
    print("=== closure cost-accuracy (Neumann) ===")
    cost = closure_cost_accuracy("neumann")

    # figure: cost-accuracy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for closure, st in cost.items():
        recs = st["records"]
        axes[0].loglog([r["bnd_Linf"] for r in recs],
                       [r["t_solve"] for r in recs], "o-", label=closure)
        axes[1].loglog([r["bnd_Linf"] for r in recs],
                       [r["nnz"] for r in recs], "o-", label=closure)
    axes[0].set_xlabel("right-boundary Linf error")
    axes[0].set_ylabel("solve time [s] (measured)")
    axes[1].set_xlabel("right-boundary Linf error")
    axes[1].set_ylabel("nnz (A)")
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.set_title("cost vs accuracy (frozen target grid)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig05_cost_accuracy.png"), dpi=300)
    plt.close(fig)

    merge_results({"exp05_solvers": {"race_N64_ghost": race,
                                     "race_N64_onesided": race2,
                                     "closure_cost": cost}})
    print("saved. figure: fig05_cost_accuracy.png")


if __name__ == "__main__":
    main()
