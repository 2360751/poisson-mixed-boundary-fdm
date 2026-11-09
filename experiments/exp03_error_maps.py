"""Experiment 3 -- error fields and boundary-error profiles.

Figures:
  fig03a_error_maps.png     -- |e| heatmaps, onesided vs ghost, N=64 (Neumann)
  fig03b_boundary_profile.png -- error along the right edge vs y, both closures
  fig03c_convergence.png    -- log-log L2 & Linf vs h for all four variants
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIGURES, merge_results, load_results

from fdm_mixed import assemble, solve_direct, to_grid, exact_grid


def main():
    N = 64
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6))
    for col, closure in enumerate(("onesided", "ghost")):
        A, b, meta = assemble(N, closure=closure, bc="neumann")
        u, _ = solve_direct(A, b)
        U = to_grid(u, meta)
        E = U - exact_grid(meta)
        ax = axes[0, col]
        im = ax.imshow(np.abs(E).T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="viridis", aspect="equal")
        ax.set_title(f"|e| , {closure} closure, N={N}, Neumann\n"
                     f"max|e| = {np.abs(E).max():.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label=r"$|u^*-u_h|$")
        if col == 0:  # show what the discretisation resolved (viz guide)
            ax.plot(np.linspace(0, 1, N + 1), np.zeros(N + 1), "w.",
                    ms=1, alpha=0.5)
            ax.set_title(f"|e| , {closure} closure, N={N}, Neumann"
                         f" (white dots: grid nodes)\n"
                         f"max|e| = {np.abs(E).max():.2e}")
        ax = axes[1, col]
        ax.semilogy(np.linspace(0, 1, N + 1), np.abs(E[N, :]) + 1e-18, "o-",
                    ms=3, label="right edge x=1")
        ax.semilogy(np.linspace(0, 1, N + 1), np.abs(E[:, N // 2]) + 1e-18,
                    "s-", ms=3, label="cut y=0.5")
        ax.set_xlabel("y (right edge) / x (cut)")
        ax.set_ylabel("|e|")
        ax.set_title("error profile")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig03_error_maps.png"), dpi=300)
    plt.close(fig)

    # convergence panel from exp02 results
    res = load_results().get("exp02_convergence", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = {"neumann_ghost": ("C0", "o", "Neumann ghost"),
              "neumann_onesided": ("C1", "s", "Neumann onesided"),
              "robin_ghost": ("C2", "^", "Robin ghost"),
              "robin_onesided": ("C3", "v", "Robin onesided")}
    for key, (c, m, lab) in styles.items():
        if key not in res:
            continue
        recs = res[key]
        hs = [r["h"] for r in recs]
        axes[0].loglog(hs, [r["L2"] for r in recs], f"{c}{m}-", label=lab)
        axes[1].loglog(hs, [r["Linf"] for r in recs], f"{c}{m}-", label=lab)
        g = recs[0]
        p2, pi = g.get("fit_L2"), g.get("fit_Linf")
        if p2 is not None:
            axes[0].annotate(f"fit p={p2:.2f}", (hs[-1], recs[-1]["L2"]),
                             textcoords="offset points", xytext=(6, 6),
                             fontsize=7, color=c)
        if pi is not None:
            axes[1].annotate(f"fit p={pi:.2f}", (hs[-1], recs[-1]["Linf"]),
                             textcoords="offset points", xytext=(6, 6),
                             fontsize=7, color=c)
    hs_ref = np.array([1 / 16, 1 / 256])
    axes[0].loglog(hs_ref, 2e-3 * hs_ref ** 2, "k--", label=r"$h^2$")
    axes[0].loglog(hs_ref, 3e-1 * hs_ref, "k:", label=r"$h$")
    axes[1].loglog(hs_ref, 4e-3 * hs_ref ** 2, "k--", label=r"$h^2$")
    axes[1].loglog(hs_ref, 4e-1 * hs_ref, "k:", label=r"$h$")
    for ax, t in zip(axes, ("discrete L2 error", "nodal Linf error")):
        ax.set_xlabel("h")
        ax.set_ylabel(t)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(t + " vs h")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig03c_convergence.png"), dpi=300)
    plt.close(fig)

    merge_results({"exp03_figs": ["fig03_error_maps.png", "fig03c_convergence.png"]})
    print("figures written:", os.listdir(FIGURES))


if __name__ == "__main__":
    main()
