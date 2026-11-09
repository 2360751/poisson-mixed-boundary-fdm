"""Five-point FDM for Direction 2 -- mixed boundary data on (0,1)^2.

    -Laplace u = f,   u = g on Gamma_D = {x=0} U {y=0} U {y=1} (incl. corners)
    Gamma_B = {1} x (0,1) carries ONE of:
        Neumann :  d_n u = q          (outward n = (+1,0), so d_n u = u_x(1,y))
        Robin   :  d_n u + beta u = r,  beta = 2
    plus a pure-Neumann depth variant on the whole boundary.

Grid:   x_i = i*h, y_j = j*h,  h = 1/N,  i,j = 0..N.
Index:  row-major in x.  Interior unknowns first, k = (i-1)*(N-1)+(j-1),
        then the right-edge unknowns (i=N, 1<=j<=N-1) are appended.
        Dirichlet nodes carry index -1 and are eliminated (moved to the RHS
        with their actual coefficient).

Boundary closures at a right-edge unknown (i=N, 1<=j<=N-1):
    "onesided"  -- first-order derivative row REPLACES the PDE row:
                    (U_N - U_{N-1})/h + beta U_N = data
    "ghost"     -- ghost value U_{N+1} eliminated, PDE row KEPT:
                    (4 U_N - 2 U_{N-1} - U_{N,j-1} - U_{N,j+1})/h^2
                        + 2*beta/h * U_N = f + 2*data/h
        (data = q for Neumann, data = r for Robin; beta = 0 for Neumann)

Sign convention: the operator is -Laplace (diagonal +4/h^2); d_n is the
OUTWARD normal derivative; Dirichlet values move to the RHS with +g/h^2.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from manufactured import (BETA, f_source, g_dirichlet, q_neumann, r_robin,
                          q_pure_west, q_pure_south, q_pure_north,
                          q_pure_east, u_star)

# ---------------------------------------------------------------------------
# index maps
# ---------------------------------------------------------------------------


def index_map(N, pure_neumann=False):
    """Return (idx, n_unknown).  idx[i,j] = k or -1 (eliminated Dirichlet)."""
    idx = -np.ones((N + 1, N + 1), dtype=int)
    k = 0
    for i in range(N + 1):
        for j in range(N + 1):
            if pure_neumann:
                idx[i, j] = k
                k += 1
            else:
                on_dirichlet = (i == 0) or (j == 0) or (j == N)
                if not on_dirichlet:
                    idx[i, j] = k
                    k += 1
    return idx, k


def grid_axes(N):
    h = 1.0 / N
    return h, np.linspace(0.0, 1.0, N + 1)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _add_entry(rows, cols, data, r, c, v):
    rows.append(r)
    cols.append(c)
    data.append(v)


def assemble(N, closure="ghost", bc="neumann", pure_neumann=False,
             ghost_west_factor=2.0):
    """Assemble A (sparse CSR), b, and metadata for the mixed-BC system.

    closure : "onesided" | "ghost"   (used on Gamma_B / all Neumann edges)
    bc      : "neumann" | "robin"    (right-edge condition for the mixed case)
    pure_neumann : if True, Neumann data on the WHOLE boundary (depth variant).
    ghost_west_factor : audit hook -- the correct ghost elimination gives a
        -2/h^2 coupling to U_{N-1,j} (factor 2).  Falsification experiment
        F2 passes 1.0 to emulate the classic 'forgot to double it' bug.
    """
    if closure not in ("onesided", "ghost"):
        raise ValueError("closure must be 'onesided' or 'ghost'")
    if bc not in ("neumann", "robin"):
        raise ValueError("bc must be 'neumann' or 'robin'")
    h, axis = grid_axes(N)
    h2 = h * h
    idx, n = index_map(N, pure_neumann=pure_neumann)
    rows, cols, data = [], [], []
    b = np.zeros(n)

    beta = BETA if bc == "robin" else 0.0

    def dirichlet_move(r, i, j):
        """Move a Dirichlet neighbour value g/h^2 to the RHS."""
        b[r] += g_dirichlet(axis[i], axis[j]) / h2

    def stencil_row(r, i, j, west, east, south, north):
        """Standard five-point row of -Laplace at (i,j).

        west/east/south/north = (coefficient_per_U, ni, nj) per neighbour;
        neighbour eliminated -> moved to RHS with its Dirichlet value.
        """
        _add_entry(rows, cols, data, r, idx[i, j], 4.0 / h2)
        b[r] += f_source(axis[i], axis[j])
        for (c_per_u, ni, nj) in (west, east, south, north):
            if idx[ni, nj] < 0:
                dirichlet_move(r, ni, nj)  # + g/h^2 * (c_per_u * h2) / h2...
                # careful: coefficient is c_per_u on the unknown; for an
                # eliminated neighbour the contribution is c_per_u * g, and
                # c_per_u = -1/h^2, so the term moved is -g/h^2 to LHS ->
                # + g/h^2 on the RHS.  dirichlet_move adds +g/h^2 which is
                # exactly -c_per_u * g when c_per_u = -1/h^2.
            else:
                _add_entry(rows, cols, data, r, idx[ni, nj], c_per_u)

    # ---- interior unknowns -------------------------------------------------
    for i in range(1, N):
        for j in range(1, N):
            if pure_neumann and (i in (0, N) or j in (0, N)):
                continue  # handled below
            r = idx[i, j]
            if r < 0:
                continue
            stencil_row(r, i, j,
                        (-1.0 / h2, i - 1, j), (-1.0 / h2, i + 1, j),
                        (-1.0 / h2, i, j - 1), (-1.0 / h2, i, j + 1))

    if not pure_neumann:
        # ---- right-edge unknowns: the two prescribed closures --------------
        for j in range(1, N):
            i = N
            r = idx[i, j]
            data_j = q_neumann(axis[j]) if bc == "neumann" else r_robin(axis[j])
            if closure == "onesided":
                # (U_N - U_{N-1})/h + beta U_N = data_j
                _add_entry(rows, cols, data, r, idx[i, j], (1.0 + beta * h) / h)
                _add_entry(rows, cols, data, r, idx[i - 1, j], -1.0 / h)
                b[r] += data_j
            else:
                # ghost-eliminated PDE row
                _add_entry(rows, cols, data, r, idx[i, j], 4.0 / h2 + 2.0 * beta / h)
                _add_entry(rows, cols, data, r, idx[i - 1, j],
                           -ghost_west_factor / h2)
                b[r] += f_source(axis[i], axis[j]) + 2.0 * data_j / h
                for nj in (j - 1, j + 1):
                    if idx[i, nj] < 0:
                        dirichlet_move(r, i, nj)   # + g/h^2
                    else:
                        _add_entry(rows, cols, data, r, idx[i, nj], -1.0 / h2)
    else:
        # ---- pure-Neumann depth variant: ghost closure on EVERY edge -------
        # x = 0 edge, 1<=j<=N-1 :  n = (-1,0),  U_{-1,j} = U_{1,j} + 2h qW
        for j in range(1, N):
            i, r = 0, idx[0, j]
            qW = q_pure_west(axis[j])
            _add_entry(rows, cols, data, r, r, 4.0 / h2)
            _add_entry(rows, cols, data, r, idx[1, j], -2.0 / h2)
            for nj in (j - 1, j + 1):
                _add_entry(rows, cols, data, r, idx[i, nj], -1.0 / h2)
            b[r] += f_source(axis[i], axis[j]) + 2.0 * qW / h
        # x = N edge, 1<=j<=N-1 :  n = (+1,0),  U_{N+1,j} = U_{N-1,j} + 2h qE
        for j in range(1, N):
            i, r = N, idx[N, j]
            qE = q_pure_east(axis[j])
            _add_entry(rows, cols, data, r, r, 4.0 / h2)
            _add_entry(rows, cols, data, r, idx[N - 1, j], -2.0 / h2)
            for nj in (j - 1, j + 1):
                _add_entry(rows, cols, data, r, idx[i, nj], -1.0 / h2)
            b[r] += f_source(axis[i], axis[j]) + 2.0 * qE / h
        # y = 0 edge, 1<=i<=N-1 :  n = (0,-1),  U_{i,-1} = U_{i,1} + 2h qS
        for i in range(1, N):
            j, r = 0, idx[i, 0]
            qS = q_pure_south(axis[i])
            _add_entry(rows, cols, data, r, r, 4.0 / h2)
            _add_entry(rows, cols, data, r, idx[i, 1], -2.0 / h2)
            for ni in (i - 1, i + 1):
                _add_entry(rows, cols, data, r, idx[ni, j], -1.0 / h2)
            b[r] += f_source(axis[i], axis[j]) + 2.0 * qS / h
        # y = N edge, 1<=i<=N-1 :  n = (0,+1),  U_{i,N+1} = U_{i,N-1} + 2h qN
        for i in range(1, N):
            j, r = N, idx[i, N]
            qN = q_pure_north(axis[i])
            _add_entry(rows, cols, data, r, r, 4.0 / h2)
            _add_entry(rows, cols, data, r, idx[i, N - 1], -2.0 / h2)
            for ni in (i - 1, i + 1):
                _add_entry(rows, cols, data, r, idx[ni, j], -1.0 / h2)
            b[r] += f_source(axis[i], axis[j]) + 2.0 * qN / h
        # four corners: TWO ghosts, each eliminated via its own edge condition
        corners = {
            (0, 0):   ((1, 0), (0, 1), q_pure_west(axis[0]),   q_pure_south(axis[0])),
            (N, 0):   ((N - 1, 0), (N, 1), q_pure_east(axis[0]), q_pure_south(axis[N])),
            (0, N):   ((1, N), (0, N - 1), q_pure_west(axis[N]), q_pure_north(axis[0])),
            (N, N):   ((N - 1, N), (N, N - 1), q_pure_east(axis[N]), q_pure_north(axis[N])),
        }
        for (i, j), (nb_x, nb_y, q_edge_x, q_edge_y) in corners.items():
            r = idx[i, j]
            _add_entry(rows, cols, data, r, r, 4.0 / h2)
            _add_entry(rows, cols, data, r, idx[nb_x[0], nb_x[1]], -2.0 / h2)
            _add_entry(rows, cols, data, r, idx[nb_y[0], nb_y[1]], -2.0 / h2)
            b[r] += f_source(axis[i], axis[j]) + 2.0 * (q_edge_x + q_edge_y) / h

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    meta = {"h": h, "N": N, "idx": idx, "n": n, "axis": axis,
            "closure": closure, "bc": bc, "pure_neumann": pure_neumann}
    return A, b, meta


# ---------------------------------------------------------------------------
# solution vector <-> grid
# ---------------------------------------------------------------------------


def to_grid(u_vec, meta):
    """Scatter the unknown vector onto the full (N+1)x(N+1) grid.

    Eliminated Dirichlet nodes are filled with the EXACT g (their discrete
    value is exact by construction), pure-Neumann nodes all have unknowns.
    """
    N, idx, axis = meta["N"], meta["idx"], meta["axis"]
    U = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            k = idx[i, j]
            if k < 0:
                U[i, j] = g_dirichlet(axis[i], axis[j])
            else:
                U[i, j] = u_vec[k]
    return U


def exact_grid(meta):
    N, axis = meta["N"], meta["axis"]
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    return u_star(X, Y)


# ---------------------------------------------------------------------------
# errors, residuals
# ---------------------------------------------------------------------------


def error_norms(u_vec, meta, shift=0.0):
    """Return dict of error norms on the FULL grid.

    L2  = h * ||e||_2  (discrete L2, same convention as test_problem_bank)
    Linf = nodal max |e|
    bnd_Linf = max |e| on the open right edge (i=N, 1<=j<=N-1)
    int_Linf = max |e| on interior nodes (1<=i,j<=N-1)
    """
    U = to_grid(u_vec, meta) + shift
    E = U - exact_grid(meta)
    h = meta["h"]
    N = meta["N"]
    out = {
        "L2": h * np.linalg.norm(E.ravel(), 2),
        "Linf": np.abs(E).max(),
        "bnd_Linf": np.abs(E[N, 1:N]).max() if not meta["pure_neumann"] else np.abs(E).max(),
        "int_Linf": np.abs(E[1:N, 1:N]).max(),
    }
    return out


def scaled_residual(A, b, u_vec):
    """Row-equilibrated residual max_i |r_i| / max_j |A_ij|.

    PDE rows scale like h^-2, first-order boundary rows like h^-1; the raw
    residual is therefore not comparable across meshes or across closures.
    The componentwise scaling removes the row units before the comparison.
    """
    r = b - A @ u_vec
    row_max = np.abs(A).max(axis=1).toarray().ravel()
    row_max[row_max == 0.0] = 1.0
    return np.abs(r / row_max).max()


def raw_residual_inf(A, b, u_vec):
    return np.abs(b - A @ u_vec).max()


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------


def solve_direct(A, b):
    lu = spla.splu(A.tocsc())
    u = lu.solve(b)
    return u, {"method": "splu", "resid_inf": raw_residual_inf(A, b, u),
               "resid_scaled": scaled_residual(A, b, u)}


def solve_pinned_direct(A, b, meta, pin_k=0):
    """Pure-Neumann gauge: replace row `pin_k` by an identity row (u=0 there),
    solve, then remove the arbitrary constant by least-squares shift."""
    A2 = A.tolil(copy=True)
    A2[pin_k, :] = 0.0
    A2[pin_k, pin_k] = 1.0
    b2 = b.copy()
    b2[pin_k] = 0.0
    u = spla.spsolve(A2.tocsr(), b2)
    return u, {"method": "splu+pinned-gauge"}


# ---------------------------------------------------------------------------
# hand-row verification (Gate: transcription / structure)
# ---------------------------------------------------------------------------


def row_defects(N, closure, bc):
    """Substitute sampled u* into each boundary-row formula; return the
    pointwise defect (truncation error) at one right-edge node.

    This is the 'verify each row by substituting sampled values of u*'
    requirement of the Problem Pack, done numerically at full precision.
    """
    y = 0.5
    h = 1.0 / N
    data_j = q_neumann(y) if bc == "neumann" else r_robin(y)
    beta = BETA if bc == "robin" else 0.0
    uN = u_star(1.0, y)
    uNm = u_star(1.0 - h, y)
    uNp = u_star(1.0 + h, y)          # conceptual ghost value of u*
    uS = u_star(1.0, y - h)
    uNb = u_star(1.0, y + h)
    fN = f_source(1.0, y)
    if closure == "onesided":
        # row: (U_N - U_{N-1})/h + beta U_N = data_j
        lhs = (uN - uNm) / h + beta * uN
        return {"lhs_minus_data": lhs - data_j, "expected_order": 1}
    # ghost row: (4 uN - 2 uNm - uS - uNb)/h^2 + 2 beta uN / h = f + 2 data/h
    lhs = (4.0 * uN - 2.0 * uNm - uS - uNb) / h**2 + 2.0 * beta * uN / h
    rhs = fN + 2.0 * data_j / h
    # also expose the centred-BC defect for the report
    centred = (uNp - uNm) / (2.0 * h)
    return {"lhs_minus_rhs": lhs - rhs,
            "centred_bc_defect": centred - (data_j - beta * uN),
            "expected_order": 2}
