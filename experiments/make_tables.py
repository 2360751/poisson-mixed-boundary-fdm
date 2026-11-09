"""Generate LaTeX table bodies from figures/results.json (report build helper)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "figures", "results.json")))
OUT = os.path.join(ROOT, "report", "tables")
os.makedirs(OUT, exist_ok=True)


def fmt(x, spec=".3e"):
    return format(x, spec)


def conv_table(key, label, caption):
    recs = R["exp02_convergence"][key]
    g = recs[0]
    lines = [f"\\begin{{table}}[H]", "\\centering\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{r" + "c" * 7 + "}",
             "\\toprule",
             "$N$ & $h$ & $\\lVert e\\rVert_{h\\ell^2}$ & $p_{h\\ell^2}$ & "
             "$\\lVert e\\rVert_{\\infty}$ & $p_{\\infty}$ & "
             "$\\lVert e\\rVert_{\\infty,\\partial}$ & $\\lVert e\\rVert_{\\infty,\\mathrm{int}}$ \\\\",
             "\\midrule"]
    p2 = g.get("pairwise_L2", [None] * len(recs))
    pi = g.get("pairwise_Linf", [None] * len(recs))
    for k, rec in enumerate(recs):
        p2s = f"{p2[k]:.3f}" if k and p2[k] else "--"
        pis = f"{pi[k]:.3f}" if k and pi[k] else "--"
        lines.append(
            f"{rec['N']} & ${rec['h']:.4f}$ & ${fmt(rec['L2'])}$ & {p2s} & "
            f"${fmt(rec['Linf'])}$ & {pis} & ${fmt(rec['bnd_Linf'])}$ & "
            f"${fmt(rec['int_Linf'])}$ \\\\")
    lines += ["\\midrule",
              f"\\multicolumn{{2}}{{l}}{{\\textbf{{fitted}}}} & "
              f"\\textbf{{{g['fit_L2']:.3f}}} & & \\textbf{{{g['fit_Linf']:.3f}}} & & "
              f"\\textbf{{{g['fit_bnd_Linf']:.3f}}} & \\textbf{{{g['fit_int_Linf']:.3f}}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def resid_table():
    recs = R["exp02_convergence"]["neumann_ghost"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\caption{Scaled algebraic residual $\\max_i |r_i|/\\max_j|A_{ij}|$ "
             "of the sparse-direct solve, and nnz of $A$ (Neumann, ghost closure). "
             "The residual stays at round-off on every mesh, far below the "
             "discretisation error in Table~\\ref{tab:conv-ghost-neu}.}",
             "\\label{tab:residuals}",
             "\\begin{tabular}{rcccc}", "\\toprule",
             "$N$ & $h$ & scaled residual & $\\lVert e\\rVert_{\\infty}$ & "
             "$\\mathrm{nnz}(A)$ \\\\", "\\midrule"]
    for rec in recs:
        lines.append(f"{rec['N']} & ${rec['h']:.4f}$ & ${fmt(rec['scaled_resid'])}$ & "
                     f"${fmt(rec['Linf'])}$ & {rec['nnz']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def rowdefect_table():
    d = R["exp01_row_defects"]
    names = [("interior", "interior five-point row", "2"),
             ("onesided_neumann", "one-sided boundary row (Neumann)", "1"),
             ("ghost_neumann", "ghost-eliminated PDE row (Neumann)", "1 (local)"),
             ("centred_bc_neumann", "centred BC condition", "2")]
    lines = ["\\begin{table}[H]", "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Pointwise row defects obtained by substituting sampled "
             "$u^*$ into each row formula (exp\\_01, measured at $h=1/64$).  "
             "Constants are compared "
             "with the Taylor predictions at $y=0.5$: one-sided "
             "$-(h/2)u^*_{xx}$, ghost row $(h/3)u^*_{xxx}$, centred BC "
             "$(h^2/6)u^*_{xxx}$, interior $-h^2(u^*_{xxxx}+u^*_{yyyy})/12$.}",
             "\\label{tab:rowdefects}",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "row & predicted leading term & fitted order & "
             "$\\text{defect}/h^{p}$ & predicted constant \\\\",
             "\\midrule"]
    preds = {"interior": ("$-u^*/6$ (with $p=2$)", f"{-2.718281828/6:.4f}"),
             "onesided_neumann": ("$-u^*/2$ (with $p=1$)", f"{-4.481689/2:.4f}"),
             "ghost_neumann": ("$u^*/3$ (with $p=1$)", f"{4.481689/3:.4f}"),
             "centred_bc_neumann": ("$u^*/6$ (with $p=2$)", f"{4.481689/6:.4f}")}
    for key, desc, order in names:
        rec = d[key]
        pred, predval = preds[key]
        h = 1 / 64
        p_exp = 1 if "order" in desc and "local" in order or order == "1" else 2
        p_exp = 1 if key in ("onesided_neumann", "ghost_neumann") else 2
        val = rec["abs_defect"][2] / (h ** p_exp)
        lines.append(f"{desc} & {pred} & {rec['fitted_order']:.3f} & "
                     f"{val:.4f} & {predval} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def solver_table():
    rows_g = R["exp05_solvers"]["race_N64_ghost"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\caption{Direct vs.\\ iterative at matched algebraic accuracy "
             "on the Neumann/ghost system at $N=64$ ($n=4033$ unknowns): every "
             "iterative run stops at relative residual $10^{-10}$; the cost "
             "measure is wall-clock solve time and iteration count (assembly, "
             "factorisation setup and plotting not counted). "
             "$\\lVert u_{\\mathrm{iter}}-u_{\\mathrm{splu}}\\rVert_{\\infty}$ "
             "is the disagreement with the direct solve (the discrete oracle).}",
             "\\label{tab:solvers}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "solver & iterations & time [s] & rel. residual & "
             "$\\lVert u-u_{\\mathrm{splu}}\\rVert_{\\infty}$ \\\\", "\\midrule"]
    for r in rows_g:
        it = str(r["iters"]) if r["solver"] != "splu" else "-- (1 factorisation)"
        name = r["solver"].replace("_", "\\_")
        lines.append(f"{name} & {it} & {r['time']:.3f} & "
                     f"${fmt(r['rel_res'])}$ & "
                     f"${fmt(r['err_vs_ref'])}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def cost_table():
    cost = R["exp05_solvers"]["closure_cost"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\caption{Cost vs.\\ accuracy for the two closures (Neumann "
             "variant).  Measured: error and wall time up to $N=512$. "
             "Extrapolated $N_{\\mathrm{need}}$ from the fitted boundary order "
             "to reach $\\lVert e\\rVert_{\\infty,\\partial}\\le\\epsilon$. "
             "Cost measure: $N$ (grid size) and $\\mathrm{nnz}(A)$; wall times "
             "are solve-only (assembly excluded).}",
             "\\label{tab:cost}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "closure & fitted $p_{\\partial}$ & $N_{\\mathrm{need}}$ "
             "($10^{-3}$) & $N_{\\mathrm{need}}$ ($10^{-4}$) & "
             "$N_{\\mathrm{need}}$ ($10^{-5}$) \\\\", "\\midrule"]
    for cl in ("ghost", "onesided"):
        st = cost[cl]
        need = st["N_needed_by_target"]
        lines.append(f"{cl} & {st['fitted_order']:.3f} & "
                     f"{need['1e-03']:.0f} & {need['1e-04']:.0f} & "
                     f"{need['1e-05']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    lines2 = ["\\begin{table}[H]", "\\centering\\small",
              "\\caption{Measured error and solve time of both closures at "
              "matched grids (Neumann).  At $N=512$ the ghost closure is "
              "$\\sim2000\\times$ more accurate at essentially the same cost "
              "-- the closure changes feasibility, not just efficiency.}",
              "\\label{tab:cost-measured}",
              "\\begin{tabular}{llrrr}", "\\toprule",
              "closure & $N$ & $\\lVert e\\rVert_{\\infty,\\partial}$ & "
              "solve time [s] & $\\mathrm{nnz}$ \\\\", "\\midrule"]
    for cl in ("ghost", "onesided"):
        for rec in cost[cl]["records"]:
            if rec["N"] in (128, 256, 512):
                lines2.append(f"{cl} & {rec['N']} & ${fmt(rec['bnd_Linf'])}$ & "
                              f"{rec['t_solve']:.3f} & {rec['nnz']} \\\\")
    lines2 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n\n" + "\n".join(lines2)


def pure_table():
    s = R["exp06_pure_neumann"]
    recs = s["convergence"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\caption{Pure-Neumann depth variant: convergence after the "
             "pinned gauge and the least-squares constant shift.  Fitted "
             f"orders: $L^2$ {s['fit_L2']:.3f}, $L^\\infty$ {s['fit_Linf']:.3f}.}}",
             "\\label{tab:pure-neumann}",
             "\\begin{tabular}{rccc}", "\\toprule",
             "$N$ & $h$ & $\\lVert e+\\bar c\\rVert_{h\\ell^2}$ & "
             "$\\lVert e+\\bar c\\rVert_{\\infty}$ \\\\", "\\midrule"]
    for rec in recs:
        lines.append(f"{rec['N']} & ${rec['h']:.4f}$ & ${fmt(rec['L2'])}$ & "
                     f"${fmt(rec['Linf'])}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    small = s["small"]
    struct = ["\\begin{table}[H]", "\\centering\\small",
              "\\caption{Pure-Neumann structure at $N=32$ ($n=1089$): right "
              "null vector, left null vector, compatibility defect, and the "
              "incompatible-RHS falsification.}",
              "\\label{tab:pure-struct}",
              "\\begin{tabular}{lr}", "\\toprule",
              "quantity & value \\\\", "\\midrule",
              f"$\\lVert A\\mathbf{{1}}\\rVert_\\infty$ & ${small['null_res']:.1e}$ \\\\",
              f"$\\lVert A^T w\\rVert_\\infty$ (smallest left singular vector) & "
              f"${small['left_defect']:.1e}$ \\\\",
              f"compatibility $|\\langle b,w\\rangle|/(\\lVert b\\rVert\\lVert w\\rVert)$ & "
              f"${small['compatibility']:.1e}$ \\\\",
              f"pinned-row defect $\\lVert Au-b\\rVert_\\infty$ (compatible $b$) & "
              f"${small['pinned_defect']:.3f}$ \\\\",
              f"pinned-row defect (falsified: $b+1$) & "
              f"${small['falsification_defect']:.1f}$ \\\\",
              f"$\\lVert e\\rVert_\\infty$ before constant shift & "
              f"${small['err_raw']:.4f}$ \\\\",
              f"$\\lVert e+\\bar c\\rVert_\\infty$ after shift & "
              f"${small['err_shifted']['Linf']:.3e}$ \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n\n" + "\n".join(struct)


def falsification_table():
    f = R["exp07_falsification"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\caption{Falsification attempts (exp\\_07): one deliberate "
             "defect per run, each destroying the second-order table of the "
             "baseline ghost closure.  The refinement study can therefore "
             "distinguish a correct implementation from a plausible-looking "
             "wrong one.}",
             "\\label{tab:falsification}",
             "\\begin{tabular}{lccc}", "\\toprule",
             "run & fitted $p_{h\\ell^2}$ & fitted $p_\\infty$ & "
             "$\\lVert e\\rVert_\\infty$ at $N=128$ \\\\", "\\midrule"]
    labels = {"baseline": "baseline (correct ghost)",
              "F1_flip_q_sign": "F1: sign of $q$ flipped",
              "F2_forget_double_west": "F2: west coupling not doubled",
              "F3_drop_top_dirichlet": "F3: top-edge Dirichlet move dropped"}
    for k in ("baseline", "F1_flip_q_sign", "F2_forget_double_west",
              "F3_drop_top_dirichlet"):
        rec = f[k]
        lines.append(f"{labels[k]} & {rec['fit_L2']:.3f} & {rec['fit_Linf']:.3f} & "
                     f"${fmt(rec['records'][-1]['Linf'])}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def cond_table():
    m = R["exp04_matrix"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\caption{Matrix structure and conditioning (exp\\_04).  "
             "Asymmetry $\\lVert A-A^T\\rVert_1/\\lVert A\\rVert_1$; condition "
             "estimates from repeated sparse solves.  The mixed assemblies are "
             "nonsymmetric (no CG); the one-sided rows worsen conditioning by "
             "$\\sim4\\times$; the pure-Neumann matrix is singular "
             "(constants in the kernel).}",
             "\\label{tab:matrix}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "assembly & $N$ & asymmetry & $\\hat\\kappa_1$ & "
             "$\\max|\\text{row sums}|$ \\\\", "\\midrule"]
    for name in ("neumann_ghost", "neumann_onesided", "robin_ghost",
                 "pure_neumann_ghost"):
        for rec in m[name]:
            if rec["N"] in (32, 128):
                lines.append(f"{name.replace('_', ' ')} & {rec['N']} & "
                             f"${rec['asymmetry']:.2e}$ & "
                             f"${rec['cond_est']:.2e}$ & "
                             f"${rec['rowsum_all_max']:.1e}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def mms_simple_table():
    """Template Table-1 shape: h, error, ratio, order (ghost/Neumann, L2)."""
    recs = R["exp02_convergence"]["neumann_ghost"]
    lines = ["\\begin{table}[H]", "\\centering",
             "\\begin{tabular}{cccc}", "\\toprule",
             "\\textbf{Mesh spacing $h$} & \\textbf{Error} & "
             "\\textbf{Ratio} & \\textbf{Order} \\\\", "\\midrule"]
    prev = None
    for r in recs:
        h, e = r["h"], r["L2"]
        if prev is None:
            lines.append(f"${h:.4f}$ & ${fmt(e)}$ & --- & --- \\\\")
        else:
            ratio = prev / e
            order = float(__import__("math").log2(ratio))
            lines.append(f"${h:.4f}$ & ${fmt(e)}$ & {ratio:.2f} & "
                         f"{order:.2f} \\\\")
        prev = e
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Convergence study for the manufactured solution "
              "$u_*$ in the discrete $L^2$ norm (ghost-point closure, "
              "Neumann data).  Ratio and order are pairwise; the fitted order "
              "over all meshes is 2.042.}",
              "\\label{tab:mms-simple}", "\\end{table}"]
    return "\n".join(lines)


def tol_table():
    s = R["exp08_tolerance_sweep"]
    rows = s["records"]
    lines = ["\\begin{table}[H]", "\\centering\\small",
             "\\begin{tabular}{crrrrr}", "\\toprule",
             "tolerance & achieved rel.\\ res. & iters & "
             "$\\lVert e\\rVert_\\infty$ (disc.) & "
             "$\\lVert u-u_{LU}\\rVert_\\infty$ & scaled res. \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(
            f"${r['tol']:.0e}$ & ${fmt(r['achieved_rel_res'])}$ & "
            f"{r['iters']} & ${fmt(r['disc_Linf'])}$ & "
            f"${fmt(r['err_vs_LU'])}$ & ${fmt(r['scaled_resid'])}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Tolerance sweep on one representative mesh "
              "($N=64$, Neumann/ghost, exp\\_08): the discretisation error "
              "stops changing for tolerance $\\le 10^{-8}$ while the "
              "algebraic disagreement with the sparse-LU oracle keeps "
              "dropping.  At tolerance $10^{-6}$ the algebraic error pollutes "
              "the refinement fit (order 0.97 instead of 2.05); at "
              "$10^{-10}$ the fitted order is the discretisation order.  "
              "Scaled row residual defined in Eq.~\\eqref{eq:scaledresid}.}",
              "\\label{tab:tolsweep}", "\\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    files = {
        "tab_conv_ghost_neu.tex": conv_table(
            "neumann_ghost", "tab:conv-ghost-neu",
            "Neumann data, ghost-point closure: four-mesh (plus confirmation) "
            "refinement in the discrete $L^2$ and nodal $L^\\infty$ norms; "
            "$p$ are pairwise observed orders.  Both norms converge at order "
            "$\\approx 2$; the boundary $L^\\infty$ error carries the maximum."),
        "tab_conv_onesided_neu.tex": conv_table(
            "neumann_onesided", "tab:conv-onesided-neu",
            "Neumann data, first-order one-sided closure: global order "
            "$\\approx 1$ in both norms, as predicted for the derivative row "
            "that replaces the PDE row."),
        "tab_conv_ghost_robin.tex": conv_table(
            "robin_ghost", "tab:conv-ghost-robin",
            "Robin data ($\\beta=2$), ghost-point closure: order $\\approx 2$; "
            "the Robin term slightly reduces the error constant."),
        "tab_conv_onesided_robin.tex": conv_table(
            "robin_onesided", "tab:conv-onesided-robin",
            "Robin data ($\\beta=2$), first-order closure: order $\\approx 1$."),
        "tab_residuals.tex": resid_table(),
        "tab_rowdefects.tex": rowdefect_table(),
        "tab_solvers.tex": solver_table(),
        "tab_cost.tex": cost_table(),
        "tab_pure.tex": pure_table(),
        "tab_falsification.tex": falsification_table(),
        "tab_matrix.tex": cond_table(),
        "tab_mms_simple.tex": mms_simple_table(),
        "tab_tolsweep.tex": tol_table(),
    }
    for fn, body in files.items():
        with open(os.path.join(OUT, fn), "w") as fh:
            fh.write(body)
        print("wrote", fn)
