"""Experiment 7 -- falsification attempts (MMS worksheet gate).

One falsification per defect class: change ONE thing and record how the
refinement table responds.  A table that does not move is a red flag for the
sensitivity of the experiment itself.

F1. Flip the sign of the Neumann datum q -> -q (ghost closure).
F2. Flip the sign of the ghost correction 2h q -> -2h q (keep q itself right).
F3. Drop the Dirichlet move on the top edge y=1 (leave g out of the RHS).

Predictions: every variant must DESTROY the second-order table of the ghost
closure (F1/F2 at the boundary rows, F3 along the whole top edge), which is
exactly the 'order 1.0 instead of 2.0' gate-4 catch.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from common import merge_results, fit_order

import fdm_mixed
from fdm_mixed import assemble, solve_direct, error_norms
import manufactured as mf


def table_with_patch(patch, ns=(16, 32, 64, 128), assemble_kw=None):
    """Run the ghost/Neumann variant with `patch()` active."""
    patch()
    recs = []
    kw = dict(assemble_kw or {})
    try:
        for N in ns:
            A, b, meta = assemble(N, closure="ghost", bc="neumann", **kw)
            u, _ = solve_direct(A, b)
            e = error_norms(u, meta)
            recs.append({"N": N, "L2": e["L2"], "Linf": e["Linf"],
                         "bnd_Linf": e["bnd_Linf"]})
    finally:
        mf.q_neumann = _orig_q
        fdm_mixed.q_neumann = _orig_q
        fdm_mixed.g_dirichlet = _orig_g
    hs = [1.0 / r["N"] for r in recs]
    p_l2, _ = fit_order(hs, [r["L2"] for r in recs])
    p_li, _ = fit_order(hs, [r["Linf"] for r in recs])
    return recs, p_l2, p_li


_orig_q = mf.q_neumann
_orig_g = mf.g_dirichlet


def f1_flip_q():
    def bad(y):
        return -_orig_q(y)
    mf.q_neumann = bad
    fdm_mixed.q_neumann = bad


def f2_forget_double_west():
    # The correct ghost elimination doubles the inward coupling
    # (-2/h^2 on U_{N-1}).  F2 emulates the classic bug: only -1/h^2.
    # Handled by passing ghost_west_factor=1.0 to assemble (audit hook),
    # so the patch is a no-op on the manufactured data.
    pass


def f3_drop_top_dirichlet():
    def bad_g(x, y):
        # top edge y=1: return 0 instead of g (Dirichlet move dropped)
        if np.isscalar(y):
            if abs(y - 1.0) < 1e-12:
                return 0.0 * x
        else:
            y = np.asarray(y)
            out = _orig_g(x, y)
            out = np.where(np.abs(y - 1.0) < 1e-12, 0.0, out)
            return out
        return _orig_g(x, y)
    mf.g_dirichlet = bad_g
    fdm_mixed.g_dirichlet = bad_g


def main():
    baseline, p0, q0 = table_with_patch(lambda: None)
    print(f"baseline (ghost, correct): L2 order {p0:.3f}, Linf order {q0:.3f}")
    out = {"baseline": {"records": baseline, "fit_L2": p0, "fit_Linf": q0}}
    cases = [("F1_flip_q_sign", f1_flip_q, None),
             ("F2_forget_double_west", f2_forget_double_west,
              {"ghost_west_factor": 1.0}),
             ("F3_drop_top_dirichlet", f3_drop_top_dirichlet, None)]
    for name, patch, kw in cases:
        recs, p_l2, p_li = table_with_patch(patch, assemble_kw=kw)
        print(f"{name:26s}: L2 order {p_l2:.3f}, Linf order {p_li:.3f}, "
              f"Linf(N=128) = {recs[-1]['Linf']:.3e}")
        out[name] = {"records": recs, "fit_L2": p_l2, "fit_Linf": p_li}
    merge_results({"exp07_falsification": out})
    print("saved to figures/results.json")


if __name__ == "__main__":
    main()
