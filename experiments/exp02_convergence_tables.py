"""Experiment 2 -- four-mesh refinement tables for all four variants.

Variants: {neumann, robin} x {onesided, ghost} closures on Gamma_B.
Meshes:   N = 16, 32, 64, 128, 256  (h = 1/N; the brief asks for h, h/2, h/4, h/8;
          N=256 is a fifth confirmation level).
Norms:    L2  = h * ||e||_2 on the full grid (test-bank convention)
          Linf = nodal max, reported globally / right boundary / interior.
Residual: row-scaled infinity residual at every level.

Pre-registered predictions:
  ghost   -> global order 2 in both norms (boundary closure does not pollute
             the global rate although its local row defect is O(h));
  onesided-> global order 1, and the boundary L-inf error dominates;
  robin   mirrors neumann with slightly smaller constants (beta=2 damps the
           right-edge mode).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from common import fit_order, pairwise_orders, merge_results

from fdm_mixed import assemble, solve_direct, error_norms, scaled_residual, to_grid, exact_grid

NS = [16, 32, 64, 128, 256]


def run_variant(closure, bc, ns=NS):
    records = []
    for N in ns:
        A, b, meta = assemble(N, closure=closure, bc=bc)
        u, info = solve_direct(A, b)
        e = error_norms(u, meta)
        sr = scaled_residual(A, b, u)
        U = to_grid(u, meta)
        E = U - exact_grid(meta)
        bnd_l2 = meta["h"] * np.linalg.norm(E[meta["N"], 1:meta["N"]])
        rec = {"N": N, "h": meta["h"], "L2": e["L2"], "Linf": e["Linf"],
               "bnd_Linf": e["bnd_Linf"], "int_Linf": e["int_Linf"],
               "bnd_L2": bnd_l2, "scaled_resid": sr,
               "raw_resid_inf": info["resid_inf"], "nnz": A.nnz}
        records.append(rec)
        print(f"  N={N:4d}  L2={e['L2']:.6e}  Linf={e['Linf']:.6e}  "
              f"bnd={e['bnd_Linf']:.6e}  int={e['int_Linf']:.6e}  "
              f"sres={sr:.2e}")
    hs = [r["h"] for r in records]
    for key in ("L2", "Linf", "bnd_Linf", "int_Linf"):
        order, dev = fit_order(hs, [r[key] for r in records])
        p_ord = pairwise_orders([r[key] for r in records])
        records[0][f"fit_{key}"] = order
        records[0][f"fitdev_{key}"] = dev
        records[0][f"pairwise_{key}"] = p_ord
        print(f"  fitted {key:9s} order = {order:.3f}  (dev {dev:.2e})  "
              f"pairwise " + ", ".join(f"{p:.3f}" if p else "-" for p in p_ord))
    return records


def main():
    out = {}
    for bc in ("neumann", "robin"):
        for closure in ("ghost", "onesided"):
            print(f"\n=== bc={bc}, closure={closure} ===")
            out[f"{bc}_{closure}"] = run_variant(closure, bc)
    merge_results({"exp02_convergence": out})
    print("\nsaved to figures/results.json")


if __name__ == "__main__":
    main()
