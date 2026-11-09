# Team O — 现场 Q&A 实战规划（Live Viva Playbook）

> 配合 `Q&A_prep.md`（技术题库 + Fast Facts）使用。这份是**作战计划**：
> 谁接哪类题、答题的固定套路、高风险题的标准答案、以及救场话术。
> 所有答题脚本为英文（现场用英文回答），规划说明为中文。

---

## 一、提问概率预测（按老师的 checklist 反推）

老师手里有 `code_review_checklist_bvp` 和 `submission_checklist_bvp`，现场提问大概率沿这两条线走。

| 概率 | 类别 | 典型问题 | 首接人 |
|---|---|---|---|
| ★★★ 必问 | 收敛阶与闭合机制 | "Why does ghost give order 2 but one-sided only order 1?" | Beixue |
| ★★★ 必问 | 验证方法 | "How do you know your results are correct?" | Beixue |
| ★★★ 必问 | 个人贡献 / AI 使用 | "What did YOU do? What did the AI do?" | 各人自己答 |
| ★★☆ 高频 | 成本结论 | "Is ghost always better? When would one-sided win?" | Zihan |
| ★★☆ 高频 | 求解器 | "Why not conjugate gradients?" | Yumo |
| ★★☆ 高频 | 纯 Neumann | "Why is your matrix singular and how did you handle it?" | Yumo |
| ★☆☆ 可能 | 代码细节 | "Show me where the ghost row is assembled." | Yumo |
| ★☆☆ 可能 | 可视化 | "Why log-log plots? What makes a plot honest?" | Zihan |
| ★☆☆ 低频 | 扩展 | "What if β=0? Variable coefficients? 3D?" | Beixue |

**分工总原则**：问题先落到"主人"（见上表），主人答第一句，其他两人可补充一个数字或一句验证，**不抢话、不重复**。

---

## 二、答题固定套路（全组统一）

每题按 **"结论 → 证据 → 数字 → 边界"** 四步答，控制在 30–45 秒：

1. **结论**（一句话，先给答案）："The short answer is..."
2. **证据**（指出处）："...and this is measured in Table 3 / exp01 of the report."
3. **数字**（背 Fast Facts 表，至少给一个）："2.04 versus 1.06."
4. **边界**（诚实说局限）："...though this is for the square domain with smooth data."

**救场话术**（不会答时）：
- 拖时间：*"Let me make sure I understand the question — you're asking about X, right?"*
- 部分会：*"I can give you the part I'm confident about: ... For the rest, [队友] may want to add."*
- 完全不会：*"That's beyond what we tested. Based on the method, my expectation would be ..., but we haven't verified it — that would be the natural next experiment."*（给出推理方向比硬猜强，老师吃这套）
- **永远不说** "I don't know" 就停 —— 永远接一句 "here is how I would find out"。

---

## 三、高风险题标准答案（重点背这 8 题）

### H1. "你们报告为什么和官方模板长得不一样？"（⚠️ 最可能被问，因为风格变了）

> **English:** "The required content is all there in the required format: the AI transparency log follows the ICS-bvp A1–A4 structure (Appendix A), the three individual contribution statements use the ICS formbox with signatures (Appendix B), and the problem definition sheet and pre-computation records are Appendix C. What we changed is the presentation: we kept the body in a classic article style, the same as our Project 1 report, so the two reports read consistently as one body of work. Content-first, decoration-second."

### H2. "What did the AI do, and what did YOU change?"（必问，且每个人都会被单独问）

按自己 ICS 里的答案答，**细节要具体**：
- **Zihan:** "I kept the AI's LaTeX table generator but rewrote the norm labels, the fitted-versus-pairwise column layout, and the caption policy — every caption interprets rather than describes."
- **Beixue:** "The AI draft had the ghost row missing a factor 2 on the west coupling. My hand derivation caught it in the row audit before any result was used — that's falsification case F2 in Table on slide 9."
- **Yumo:** "I replaced the AI's iterate-difference stopping rule with the relative-residual criterion, added the zero-initial-residual guard, and vectorised the Jacobi update."
- 万能收尾：*"Everything the AI produced went through the claim-audit protocol in Appendix A — prediction first, audit before use."*

### H3. "Why does the ghost row have an O(h) local defect but global order 2?"（核心概念，全组必会）

> "Both rows have O(h) local defects — that's the trap. The difference is *where the defect lives*. The one-sided row replaces the PDE row, so its defect IS the boundary error, and the global order is 1. The ghost row's defect sits in a boundary layer of width h — the two-point normal difference constrains e_N minus e_{N-1} to O(h³) — so the global effect is O(h²). Local order doesn't determine global order; geometry does."

### H4. "Is the ghost closure ALWAYS better?"（设坑题，答案是否定的）

> "For this problem class — smooth solution, right-angle boundary, second-order data — yes, by a wide margin: N of 47 versus 8600 at the 1e-4 target. But we should be careful: at very coarse meshes the two-point normal difference through the boundary is less accurate relative to the interior stencil's local behaviour, and if the solution had a boundary singularity, both closures degrade — our limitation section says non-right-angle corners are untreated. The claim is about *this* benchmark, measured — not a theorem for all PDEs."

### H5. "Why not conjugate gradients? The Laplacian is self-adjoint."（概念陷阱）

> "The *continuous* operator is self-adjoint, but the *discrete* matrix isn't: the ghost substitution doubles the west coupling of edge rows, and the one-sided row has the (3,−4,1)/2h pattern. Measured ‖A−Aᵀ‖/‖A‖ is 0.02 for ghost and 0.15 for one-sided. CG on a nonsymmetric matrix has no convergence guarantee, so we use sparse LU as reference and BiCGSTAB+ILU — 39 iterations, 0.049 s at N=64."

### H6. 追问 ICS 里自己写的 "one thing I still do not fully understand"（老师读过 ICS 会打这个点）

- **Zihan（L∞ 1.84）:** "I can attribute it — the corner rows carry the maximum error and their defect is first-order in the tangential direction — but I can't yet derive the rate. I'd bound it via the corner-row defect and confirm with a corner-refined experiment; that's stated as our main limitation."
- **Beixue（5.8e-6 兼容性缺陷）:** "The corner nodes sample two one-sided normal data, and the boundary integral is trapezoid-sampled, so the discrete compatibility ⟨b, w⟩=0 holds only to the quadrature error. I can bound it, not yet expand it term by term."
- **Yumo（ILU 39 次迭代 vs SOR 3.7e3）:** "ILU approximates A⁻¹ directly so BiCGSTAB needs few iterations; SOR only damps error modes with the scalar ω, and the h⁻² condition number means thousands of sweeps. Predicting the exact counts would need an eigenvalue analysis — I use the preconditioner, I can explain the principle, not the number."
- **关键态度**：答到"我能 bound / 我能归因 / 我知道下一步怎么算"就停，不要现编。

### H7. "Gauss–Seidel should be faster than Jacobi — your table says otherwise. Explain."（实测负结果）

> "In iteration count GS wins, but each GS sweep is a triangular solve — sequential, with row access overhead — while Jacobi is one vectorised mat-vec. On this machine, at N=64, GS costs more per iteration than it saves. It's a wall-time result on one machine, and we report it as measured rather than tuning it away; the closure ranking doesn't depend on the solver anyway."

### H8. "One command reproduces everything — prove it."（演示题）

> "In the code zip: `python run_all.py` runs exp01 through exp08 plus the smoke tests — about 80 seconds — and regenerates every figure and the results.json that every table body is generated from. The report cannot drift from the code because the tables aren't typed by hand."（如被要求现场跑：`python code/run_all.py`，确认已装 numpy/scipy/matplotlib。）

---

## 四、按主题的中概率题速答卡（10–20 秒版）

| 问题 | 一句话答案 |
|---|---|
| Why u*=e^{x+y}? | 全阶导数有界 → Taylor 常数能手算；数据闭式；五条测试性质全占 |
| What happens at corners? | 混合问题：四角全归 Dirichlet，无歧义；纯 Neumann：角点采两条法向数据 → 兼容性缺陷来源 |
| Why the scaled residual ρ = max\|r\|/max\|A_ij\|? | 原始残差随 h² 缩放不可跨网格比较；scaled 版网格无关（exp05） |
| Tolerance policy? | rel-residual，tol ~ C·h²，让代数误差永远低于离散误差；tol=1e-6 会把拟合阶污染到 0.97（exp08） |
| What if β=0? | 退化为 Neumann：ghost 关系里 2β/h 对角项消失，公式自动覆盖（代码同一分支） |
| What if f changes? | 装配不变，只换 RHS；manufactured 框架下可任意换 u* 重测 |
| Why row-major ordering? | 只影响索引映射不影响数学；CSR 需要 — 注释写在 fdm_mixed.py 头部 |
| How measure κ? | solve-based 一范数探针（非证明），growth h⁻²·⁰⁰ vs h⁻¹·⁹⁷，one-sided 差约 4 倍 |
| Is your error norm standard? | 离散 L² = h·(Σe²)^½，L∞ 逐节点 max；Dirichlet 消元节点误差为 0；右边界/内部 max 分开报 |
| Why no adaptivity/multigrid? | 超出 brief；AMG 是 natural future work（报告 Limitations 已写） |

---

## 五、现场流程与心理准备

1. **老师提问时**：听完再答（别打断），拿不准就复述确认："So the question is about X?"
2. **谁答第一句**：按第一节表格。主人不在/卡壳，任何人补位，格式："To add to what Beixue said..."。
3. **数字脱口而出**（Fast Facts 必背 6 个）：2.04/1.06、47 vs 8592、1.482 vs 1.494、5.8e-6、39 iters/0.049s、2.052 vs ≤0.04（证伪）。
4. **被问到没做的东西**：统一句式 "we haven't tested that — our expectation is X because Y, and it would be the natural next experiment"。**不辩解、不装懂**。
5. **每人自己准备 30 秒自我介绍**（贡献一句话版），如果老师说 "introduce your contribution first"。
6. **如需现场展示代码**：提前打开 `code/fdm_mixed.py` 并定位 ghost row 函数（`_edge_rows` 附近），别现场翻找。

---

## 六、三个"自杀回答"（全组禁用）

- ❌ "That's just what the AI wrote." —— 正确版：H2 的个人改动答案。
- ❌ "The report was formatted by our tool." —— 正确版：H1。
- ❌ "It works, we don't know why." —— 正确版：给推理方向 + 承认未验证（第二节救场话术）。
