# Team O — Project 2 Viva / Q&A Preparation Sheet

> For the video presentation Q&A. Every member should be able to answer
> *any* question below, including outside their own part. Answers cite the
> report section or experiment script where the evidence lives.

---

## A. Model & problem definition (anyone)

**Q: Why this specific boundary configuration (Direction 2)?**
A: The brief's Direction 2 asks for mixed data with *two* second-order-look
closures on a Neumann/Robin edge. It is exactly the setting where "the
boundary closure is part of the method" is demonstrable, and it matches our
Weeks 5–8 accumulated work (ghost-point elimination vs. one-sided
difference). We add the Robin variant and the pure-Neumann depth item as the
new work of Project 2.

**Q: Why the manufactured solution $u^\ast=e^{x+y}$?**
A: Harmonic-adjacent simplicity: $f=-2e^{x+y}$ is non-trivial but smooth, all
boundary data are available in closed form ($g=e^{x+y}$, $q=e^{1+y}$,
$r=3e^{1+y}$ with $\beta=2$), and derivatives of every order are bounded on
$\bar\Omega$, so Taylor-defect constants are easy to compute by hand
(Appendix problem definition sheet). The test bank's five properties all
hold: exact solution known; mixed BC; a Neumann edge; corner treatment; and
compatibility is *not* automatic in the pure-Neumann variant (that is the
point of the depth item).

**Q: What happens at the four corners?**
A: Corner values are Dirichlet data in the mixed problem ($g$ known at all
four corners), so corners are eliminated into the RHS. In the pure-Neumann
variant corners sample *two* one-sided normal data; that is the main source
of the $5.8\times10^{-6}$ compatibility defect and of the $L^\infty$ order
1.84 (bounded, not fully derived — we say so in the report §Conclusions).

## B. Discretisation & closures (Beixue Liang leads; all must know)

**Q: Write the ghost row on $x=1$ for Neumann.**
A: PDE row at $(x_i,y_j)$ with $x_i=1$: $(4u_{i,j}-u_{i-1,j}-u_{i+1,j}
-u_{i,j-1}-u_{i,j+1})/h^2 = f$. Ghost relation from
$(u_{i+1,j}-u_{i-1,j})/(2h)=q_j$: $u_{i+1,j}=u_{i-1,j}+2h\,q_j$. Substituting:
the west coupling becomes $-2/h^2$, diagonal $4/h^2$, RHS gets
$+f_{i,j}+2q_j/h$. (Eqs. 8–9 of the report; implemented in
`code/fdm_mixed.py`, row function `_edge_rows`.)

**Q: And for Robin?**
A: Ghost relation: $u_{i+1,j}=u_{i-1,j}+2h(r_j-\beta u_{i,j})$. The
substitution adds $+2\beta/h$ to the diagonal and $+2r_j/h$ to the RHS —
diagonal becomes $4/h^2+2\beta/h$. (Report Eq. 10–11.)

**Q: Why does the one-sided row only give order 1?**
A: Its local truncation defect is $-(h/2)u^\ast_{xx}+O(h^2)$ — order $h$ and
it sits *on* the boundary line where the solution error is measured. The
defect is not confined to a thin layer, so the global boundary error inherits
order $h$. The ghost row's defect $(h/3)u^\ast_{xxx}$ is also $O(h)$ but the
boundary-layer analysis (Week 8, the $e_N-e_{N-1}=O(h^3)$ constraint) shows
the global effect is $O(h^2)$.

**Q: What is the fitted order of the ghost row defect in exp01?**
A: 0.992 (expected 1.0 for $p=1$), and the measured constant 1.482 matches
the hand-predicted $u^\ast_{xxx}/3=1.494$ at $y=0.5$ to three digits. The
*global* solution order is 2.04 — the distinction between local defect
order and global convergence order is the key conceptual point of the
project (all three members were asked this in their ICS).

## C. Verification & falsification (Beixue Liang leads)

**Q: What exactly does exp01 check?**
A: It substitutes sampled $u^\ast$ into each row *formula* (not solving any
system) and fits defect$/h^p$ against the hand-derived Taylor constant:
interior $-h^2(u_{xxxx}+u_{yyyy})/12$, one-sided $-(h/2)u_{xx}$, ghost
$+(h/3)u_{xxx}$, centred BC $+(h^2/6)u_{xxx}$. Four rows, four matches to
3 digits (report Table 6). This catches assembly errors before any solver runs.

**Q: How do you know the refinement table can't be fooled by a wrong code?**
A: exp07 injects three plausible defects: (F1) flip sign of $q$, (F2) forget
to double the west coupling in the ghost row, (F3) drop the top-edge
Dirichlet elimination. Orders collapse to ≈0, −0.08, 0.04 respectively
(baseline 2.05), and $N=128$ errors blow up from $1.4\times10^{-5}$ to
3.5–7.4. Each defect fails *distinctively* — the study is sensitive not just
noisy.

**Q: Why report scaled residual $\max|r_i|/\max|A_{ij}|$?**
A: It is mesh-size independent: raw residuals shrink like $h^2$ just because
the equations scale like $1/h^2$. The scaled residual keeps algebraic error
comparable across grids (exp 05 stopping rule: tol $\sim C h^2$ so algebraic
error never pollutes discretisation error — Week 7 policy).

## D. Implementation (Yumo Li leads)

**Q: How is the unknown laid out?**
A: Row-major map of interior nodes only: $i=1..N-1$, $j=1..N-1$, flattened
as $k = i(N-1)+j$ (NumPy `reshape` consistent). Dirichlet values never
become unknowns — they are eliminated into $b$. Sparse CSR assembly with
`scipy.sparse`; direct solves via `spsolve`/`splu`.

**Q: Why is the matrix nonsymmetric?**
A: Mixed boundary conditions break the symmetry of the five-point Laplacian:
the ghost substitution doubles the west coupling of edge rows ($-2/h^2$
instead of $-1/h^2$), and the one-sided row's coefficients $(3,-4,1)/2h$
are not a symmetric pattern. Measured $\|A-A^T\|/\|A\|$: 0.02 (ghost) vs
0.15 (one-sided) at $N=64$.

**Q: Which solver should one use?**
A: For these sizes, BiCGSTAB+ILU: at $N=64$ it matches splu to
$6.9\times10^{-9}$ in 39 iterations / 0.049 s — faster than splu at larger
$N$ where fill-in dominates. Krylov methods are also the only feasible
option past $n\sim10^6$. Jacobi/GS/SOR converge but need $10^3$–$10^4$
iterations.

**Q: Why tolerance $\sim C h^2$?**
A: If the algebraic error is $O(h^2)$ it sits at the discretisation level
and does not pollute measured convergence orders; solving tighter wastes
time, looser destroys the table (we verified with the race experiment —
rel. residual $10^{-10}$ keeps err vs splu $<10^{-7}$).

## E. Pure-Neumann depth item (all)

**Q: Why is the pure-Neumann matrix singular?**
A: Neumann data on the whole boundary $\Rightarrow$ solution unique up to an
additive constant $\Rightarrow$ $A\bm 1 = 0$ exactly (all row sums zero;
verified to machine zero in exp06). Smallest left singular vector is the
constant too ($2.6\times10^{-12}$ defect), so $b$ must be orthogonal to it
(compatibility).

**Q: Why is compatibility not exactly satisfied discretely?**
A: Corner rows sample two one-sided normal data and the boundary integral is
trapezoid-sampled: defect $5.8\times10^{-6}$. We report it instead of
silently projecting $b$. Recovery: pin $u(0,0)=0$ (gauge), then least-squares
constant shift before measuring error. Orders survive: $L^2$ 2.05,
$L^\infty$ 1.84.

**Q: What is the falsification for this variant?**
A: An incompatible RHS (add a constant to $b$ that breaks $\langle b,w\rangle=0$):
the pinned system still solves but the error plateau proves the data are
inconsistent — the audit trail shows the check *can* fail.

## F. Cost / feasibility (Zihan Xu leads)

**Q: The "feasibility" claim — quantify it.**
A: Target boundary $L^\infty \le 10^{-4}$: ghost needs $N\approx47$
($n\approx2.2\times10^3$); one-sided needs $N\approx8.6\times10^3$
($n\sim7\times10^7$) — sparse direct LU is infeasible at that size (fill-in;
our Week 8 runs hit OOM at comparable sizes). At matched $N=512$ and matched
wall time (≈2.7 s) ghost reaches $8.5\times10^{-7}$, one-sided only
$1.7\times10^{-3}$.

**Q: Do timings include assembly?**
A: Table 8 reports solve time only (assembly/factorisation-setup/plotting
excluded) and says so in the caption; the cost figure additionally records
$\mathrm{nnz}$ and full assembly+solve times in `results.json`.

## G. AI transparency & process (Zihan Xu leads)

**Q: What did the AI actually do and what did you change?**
A: Three logged claim audits (Report Appendix A): A2-row-1 — table generator kept
but norm labels, fitted-vs-pairwise columns and caption policy rewritten;
C2 — assembly skeleton kept but ghost-row west coefficient (missing factor
2) corrected and RHS collection rewritten; C3 — solver-driver skeleton kept
but stopping rule replaced by the $Ch^2$ policy. Five audit records
(including one rejected output) in the same appendix. Each member's ICS
(Appendix B) states one AI item they changed and one concept they still find
difficult.

**Q: What is still open?**
A: (i) Full derivation of the pure-Neumann $L^\infty$ rate 1.84 (corner
quadrature attribution only); (ii) non-right-angle corners; (iii) SOR
$\omega$ tuning / AMG comparison. Listed in report §Conclusions and slide 14.

---

### Fast facts (memorise)

| quantity | value |
|---|---|
| fitted orders ghost (Neumann) | $L^2$ 2.04, $L^\infty$ 2.00 |
| fitted orders one-sided (Neumann) | $L^2$ 1.06, $L^\infty$ 1.02 |
| ghost row Taylor constant measured | 1.482 vs predicted 1.494 |
| compatibility defect (pure N.) | $5.8\times10^{-6}$ |
| $\|A\bm1\|_\infty$ | 0 (machine zero) |
| bicgstab+ILU @ N=64 | 39 iters, 0.049 s, $6.9\times10^{-9}$ vs splu |
| target $10^{-4}$: ghost vs one-sided $N$ | 47 vs 8592 |
| baseline falsification table | 2.052 (correct) vs ≤0.04 (broken) |
