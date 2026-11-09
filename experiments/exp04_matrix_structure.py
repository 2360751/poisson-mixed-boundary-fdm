"""Experiment 4 -- matrix structure: spy, nonsymmetry, conditioning trend.

Pre-registered predictions:
  M1. Dirichlet-only five-point assembly is SPD (symmetric); the mixed
      assemblies are NOT symmetric: ||A-A^T||/||A|| > 0, with the ghost
      assembly's asymmetry from the doubled inward coupling and the
      onesided assembly's from the h^-1-scaled derivative row.
  M2. Row sums vanish on interior rows (constants in the kernel of the
      stencil), but NOT on the first-order boundary rows -- quantify both.
  M3. Conditioning grows like O(h^-2) for the ghost assembly (as for the
      plain Laplacian) and is worse for the onesided assembly because of
      the mixed row scaling.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from common import FIGURES, merge_results

from fdm_mixed import assemble


def asymmetry(A):
    D = (A - A.T)
    return abs(D).sum() / abs(A).sum()


def cond_estimate(A):
    """1-norm condition estimate via splu solves on A and A^T."""
    n = A.shape[0]
    lu = spla.splu(A.tocsc())
    # 1-norm estimator: solve A y = x, A^T z = y with random-ish x
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = lu.solve(x)
    z = lu.solve(x, trans="T")
    # crude power iteration for ||A^{-1}||_1 proxy: a few solves
    v = rng.standard_normal(n)
    for _ in range(12):
        v = lu.solve(v)
        nv = np.linalg.norm(v, 1)
        if nv == 0:
            break
        v = v / nv
    ainv1 = np.linalg.norm(lu.solve(v), 1)
    a1 = abs(A).sum(axis=0).max()
    return float(a1 * ainv1)


def main():
    Ns = [16, 32, 64, 128]
    rows = {}
    for name, kw in [("neumann_onesided", dict(closure="onesided", bc="neumann")),
                     ("neumann_ghost", dict(closure="ghost", bc="neumann")),
                     ("robin_ghost", dict(closure="ghost", bc="robin")),
                     ("pure_neumann_ghost", dict(closure="ghost", pure_neumann=True))]:
        stats = []
        for N in Ns:
            A, b, meta = assemble(N, **kw)
            asy = asymmetry(A)
            # row sums
            rs = np.asarray(A.sum(axis=1)).ravel()
            interior = np.abs(rs[meta["idx"][1:N, 1:N].ravel()]).max()
            allmax = np.abs(rs).max()
            c = cond_estimate(A)
            stats.append({"N": N, "nnz": A.nnz, "asymmetry": asy,
                          "rowsum_interior_max": float(interior),
                          "rowsum_all_max": float(allmax), "cond_est": c})
            print(f"  {name:18s} N={N:4d}  nnz={A.nnz:7d}  "
                  f"||A-A^T||/||A||={asy:.3e}  rowsum_int={interior:.1e}  "
                  f"rowsum_all={allmax:.1e}  cond~{c:.2e}")
        hs = [1.0 / s["N"] for s in stats]
        from common import fit_order
        p, _ = fit_order(hs, [s["cond_est"] for s in stats])
        print(f"  -> cond(h) fitted order {p:.2f} (expect ~ -2)")
        rows[name] = stats

    # spy plots, N=32
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (name, kw) in zip(axes, [
            ("neumann onesided", dict(closure="onesided", bc="neumann")),
            ("neumann ghost", dict(closure="ghost", bc="neumann")),
            ("pure Neumann ghost", dict(closure="ghost", pure_neumann=True))]):
        A, b, meta = assemble(32, **kw)
        ax.spy(A, markersize=1.2)
        ax.set_title(f"{name}\nN=32, nnz={A.nnz}, asym={asymmetry(A):.1e}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig04_spy.png"), dpi=300)
    plt.close(fig)

    merge_results({"exp04_matrix": rows})
    print("saved. figure: fig04_spy.png")


if __name__ == "__main__":
    main()
