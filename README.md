# Team O — Project 2: Mixed Boundary Data for the Poisson BVP

Numerical Solution of Elliptic Boundary Value Problems (MTH321, Weeks 5–9).
**Direction 2 — mixed boundary data (Dirichlet / Neumann / Robin)** on the
unit square, five-point FDM, with the two boundary closures prescribed by the
Problem Pack.

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib
python code/run_all.py            # regenerates ALL report figures and tables
```

Runtime: a few minutes (the cost–accuracy sweep includes sparse LU at
N=512; the solver race runs many-iteration Jacobi sweeps at N=64).
`python tests/test_smoke.py` runs the fast pre-registered sanity checks.

Dependencies (requirements.txt): `numpy`, `scipy`, `matplotlib`.
The course FEM training stack (`scikit-fem`, `pyamg`) is used only by the
tutorial demo `fem_amg_demo.py` (not part of this project's pipeline); the
environment-setup versions we verified against are skfem 12.0.2 / pyamg 5.4.1.

## Headline results (all produced by `code/run_all.py`)

| Quantity | Value |
|---|---|
| Manufactured solution | `u* = e^{x+y}`, `f = -2e^{x+y}`, `q = e^{1+y}`, `r = 3e^{1+y}` (β = 2) |
| Ghost closure, fitted order (L2 / L∞) | 2.04 / 2.00 (Neumann), 2.05 / 2.00 (Robin) |
| One-sided closure, fitted order (L2 / L∞) | 1.06 / 1.02 (Neumann), 1.05 / 1.01 (Robin) |
| Hand-row defect, one-sided row | `-(h/2) u*_xx` — measured constant matches Taylor to 3 digits |
| Hand-row defect, ghost-eliminated PDE row | `(h/3) u*_xxx + O(h^2)` — local O(h), global order 2 |
| Pure-Neumann depth | `A @ 1 = 0` exactly; compat. defect 5.8e-6 (reported); pinned gauge + mean shift → L2 order 2.05 |
| Cost to reach boundary L∞ ≤ 1e-4 | ghost: N ≈ 47 (n ≈ 2.2e3); one-sided: N ≈ 8.6e3 (n ≈ 7e7, infeasible for sparse LU) |
| Tolerance sweep (N=64) | disc. error plateaus at 5.4e-5 for tol ≤ 1e-8; at tol 1e-6 the fitted order degrades to 0.97 |
| Matrix nonsymmetry ‖A−Aᵀ‖/‖A‖ | 0 for Dirichlet-only; > 0 for both mixed assemblies → no CG |

## Repository layout

```
code/run_all.py          reproduce every figure and every quoted number
code/manufactured.py     Direction-2 manufactured data + analytic self-checks
code/fdm_mixed.py        assembly (both closures, Neumann/Robin, pure-Neumann),
                         index map (documented in header), norms,
                         scaled residuals, hand-row defect probe
code/solvers_iterative.py  Jacobi / Gauss-Seidel / SOR / BiCGSTAB+ILU
experiments/exp01..08    row verification, refinement tables, error maps,
                         matrix structure, solver & cost-accuracy,
                         pure-Neumann depth, falsification, tolerance sweep
experiments/make_tables.py  generates every report table from figures/results.json
tests/test_smoke.py      pre-registered sanity checks (orders, null space)
requirements.txt         pinned numpy/scipy/matplotlib
figures/                 regenerated PNGs (dpi=300) + results.json
report/                  LaTeX report (main.tex → main.pdf)
slides/                  presentation deck (Team O.pdf)
```

## How to reproduce each report figure

| Figure | Script | Output |
|---|---|---|
| Fig 1 (error maps + profiles) | `experiments/exp03_error_maps.py` | `figures/fig03_error_maps.png` |
| Fig 2 (convergence, 4 variants) | `experiments/exp03_error_maps.py` | `figures/fig03c_convergence.png` |
| Fig 3 (tolerance sweep) | `experiments/exp08_tolerance_sweep.py` | `figures/fig06_tolerance_sweep.png` |
| Fig 4 (cost vs accuracy) | `experiments/exp05_solvers_cost.py` | `figures/fig05_cost_accuracy.png` |
| Fig 5 (sparsity patterns) | `experiments/exp04_matrix_structure.py` | `figures/fig04_spy.png` |
| All tables | `experiments/make_tables.py` | `report/tables/*.tex` |

Figure files are named by role (no timestamps); the report cannot drift
from the code because every table body is generated from
`figures/results.json`.

## Conventions (stated in code comments)

- Sign convention: operator is $-\Delta$ (positive diagonal); $\partial_n$
  is the outward normal, so on $x=1$, $\partial_n u = u_x(1,y)$.
- Index map: row-major in $x$; interior unknowns $k=(i-1)(N-1)+(j-1)$,
  right-edge unknowns appended; Dirichlet nodes eliminated to the RHS
  (see `code/fdm_mixed.py` header).
- Sparse assembly directly into COO/CSR; no dense fallback.
- Iterative solvers stop on the residual, never the iterate difference;
  tolerance policy $\mathrm{tol} \sim C h^2$ (algebraic error below
  discretisation error).
- Random elements (condition estimator probe) use a fixed seed.
