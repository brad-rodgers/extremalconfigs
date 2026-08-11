# Large-\(N\) numerical investigation of extremal gap profiles

## Scope

This package studies the spectral radius of the real skew-symmetric matrix

\[
B_{mn}=\frac{\sqrt{\delta_m\delta_n}}{\lambda_m-\lambda_n},\qquad m\ne n,
\]

where \(g_j=\lambda_{j+1}-\lambda_j>0\),
\(\delta_1=g_1\), \(\delta_N=g_{N-1}\), and
\(\delta_k=\min(g_{k-1},g_k)\) for interior indices.  The objective is the
largest eigenvalue of the Hermitian matrix \(iB\), which equals the spectral
radius of \(B\).

The sparse sequence investigated is

\[
N=48,800,1000,1700,3000,5000,8000,12000,20000,50000.
\]

The largest fully integer-count-refined profile is at \(N=20000\).  The
\(N=50000\) profile is an optimized, directly validated admissible candidate,
but its integer plateau counts were initialized from a pi-geometric profile
and were not exhaustively refined.  It is therefore strong evidence for the
spectral-radius behavior and gap-level scales, but not an independent test of
the proposed plateau-count constants.

## Normalization and the meaning of \(\gamma_0\)

The conjecture in the prompt writes \(\gamma_0=0\), but a literal zero gap
would make two consecutive \(\lambda\)'s equal.  In view of the preceding
minimum-gap normalization, this investigation interprets that statement as

\[
\log \gamma_0=0,\qquad\text{equivalently}\qquad \gamma_0=1.
\]

All reported gap profiles have minimum gap exactly one.

## Structural assumptions

The large-\(N\) search is compressed rather than unrestricted.  It assumes:

1. reversal symmetry of the gap list;
2. a contiguous piecewise-constant profile whose levels increase outwards,
   written
   \[
   \gamma_L^{[n_L]},\ldots,\gamma_1^{[n_1]},1^{[n_0]},
   \gamma_1^{[n_1]},\ldots,\gamma_L^{[n_L]};
   \]
3. positive, strictly increasing levels \(1=\gamma_0<\gamma_1<\cdots<\gamma_L\).

The inequality \(n_{j+1}\le n_j\) was **not** imposed in the principal runs;
it emerged in every retained profile.  Conjectures (4)--(6) were not imposed
as optimization constraints.  Pi-geometric count lists were used as some
high-\(N\) starting points, and records for which this prevents an independent
test of the count conjectures are explicitly marked in the data files.

These assumptions make \(N=50000\) feasible, but they also mean that the
results are not a global search over all \(N-1\) gaps and are not a proof of
symmetry or plateau structure.

## Numerical method

### Matrix-free Hermitian operator

A dense \(N\times N\) matrix is impossible to store economically at
\(N=50000\).  The program applies the Cauchy kernel \(1/(x-y)\) with a
one-dimensional hierarchical block decomposition:

- nearby blocks are evaluated directly;
- well-separated blocks use a source-centered Taylor expansion;
- the reverse block is defined as the negative transpose of the same
  approximation, preserving skew-symmetry exactly in floating point.

The resulting matrix-vector product costs approximately
\(O(pN\log N)\), where \(p\) is the Taylor rank.  A warm-start Lanczos method
then computes the largest eigenpair of \(iB\).

### Continuous level optimization

The levels are parameterized by positive log-ratios

\[
t_j=\log(\gamma_j/\gamma_{j-1})>0.
\]

At very large dynamic range, a formally analytic squared-kernel gradient
suffers cancellation.  The final program therefore estimates derivatives of
the Rayleigh quotient at the current eigenvector by centered finite
differences and uses damped diagonal-Newton updates with line search.

### Integer plateau-count search

For a fixed number of levels, integer counts are varied by transferring gaps
across adjacent plateau boundaries.  Promising moves are followed by
continuous level repolishing.  Outer split and merge branches can also be
tried.  This is a local discrete search, not an enumeration of all integer
compositions.

### Numerical validation

Several independent checks are included:

- dense eigensolves at \(N=800\) and \(N=1000\);
- stricter hierarchical ranks and separation thresholds;
- blockwise Taylor truncation bounds for Rayleigh quotients;
- direct \(O(N^2)\) matrix-vector products at \(N=8000\) and \(N=50000\);
- a separate NumPy-only all-pairs verifier that does not import the
  hierarchical solver.

For the main \(N=8000\) witness, the independent all-pairs sum in extended
precision gives

\[
v^*(iB)v=3.1416093314529186795
       =\pi+1.66778631257\times10^{-5}.
\]

The sum of the absolute pair contributions is only about 1.228 times the final
Rayleigh quotient, so this comparison is not the result of severe global
cancellation.  At \(N=50000\), the direct all-pairs value is

\[
3.142844355775986379
   =\pi+0.00125170218619.
\]

These are numerical checks rather than interval-arithmetic proofs, but the
margins above \(\pi\) are many orders of magnitude larger than the observed
cross-method discrepancies.

## Main numerical results

Counts are listed from the central plateau outwards:
\([n_0,n_1,\ldots,n_L]\).

| \(N\) | candidate \(\rho(B)\) | \(\rho(B)-\pi\) | \(L\) | \(n_0/N\) | counts | count status |
|---:|---:|---:|---:|---:|---|---|
| 48 | 2.975825380668484 | -0.165767273 | 2 | 0.854167 | [41, 2, 1] | unrestricted small-\(N\) search |
| 800 | 3.132299222002076 | -0.009293432 | 5 | 0.801250 | [641, 54, 16, 6, 2, 1] | locally count-refined |
| 1000 | 3.134307741092329 | -0.007284912 | 5 | 0.799000 | [799, 69, 20, 7, 3, 1] | locally count-refined |
| 1700 | 3.137588669078718 | -0.004003985 | 6 | 0.791176 | [1345, 125, 34, 11, 4, 2, 1] | locally count-refined |
| 3000 | 3.139666212413383 | -0.001926441 | 6 | 0.747000 | [2241, 274, 71, 22, 8, 3, 1] | locally count-refined |
| 5000 | 3.140865817027155 | -0.000726837 | 7 | 0.688600 | [3443, 575, 141, 41, 14, 5, 1, 1] | locally count-refined |
| 8000 | 3.141609331452917 | +0.000016678 | 7 | 0.655125 | [5241, 983, 274, 81, 27, 10, 3, 1] | locally count-refined and directly validated |
| 12000 | 3.142054933607138 | +0.000462280 | 7 | 0.681917 | [8183, 1302, 414, 132, 42, 13, 4, 1] | pi-geometric count seed; levels optimized |
| 20000 | 3.142448640354002 | +0.000855987 | 8 | 0.681550 | [13631, 2122, 739, 221, 69, 22, 7, 3, 1] | locally count-refined |
| 50000 | 3.142844355775985 | +0.001251702 | 9 | 0.681580 | [34079, 5425, 1727, 550, 175, 56, 18, 6, 2, 1] | pi-geometric count seed; levels optimized and directly validated |

The complete levels, expanded gaps, residuals, and search-status metadata are
in the JSON, CSV, and NPZ files.

## Assessment of the six conjectures

### (1) Symmetry

No new independent high-\(N\) test was performed because symmetry was imposed,
as authorized in the prompt.  The unrestricted calculations through \(N=48\)
remain the independent numerical evidence for this conjecture.

### (2) Symmetric plateau form, increasing levels, decreasing counts

The symmetric contiguous plateau form and increasing levels were imposed.  The
count inequalities \(n_{j+1}\le n_j\) were not imposed and held for every
reported candidate.  Thus the count-monotonicity portion receives additional
conditional support, while the plateau form itself is not independently tested
at large \(N\).

### (3) Number of levels

If `log` means the natural logarithm, the proposed statement
\(L_N=\log N+O(1)\) is not compatible with conjecture (6).  Geometric counts
with ratio \(\pi\) predict instead

\[
\boxed{L_N=\log_\pi N+O(1).}
\]

The observed offsets \(L_N-\log_\pi N\) stay bounded, approximately between
\(-1.21\) and \(-0.44\) for the large-\(N\) records.  A sharper empirical rule
is obtained with

\[
c=\frac12\left(1-\frac1\pi\right):
\qquad
L_N\approx \operatorname{nint}\!\bigl(\log_\pi(cN)\bigr).
\]

This nearest-integer rule matches every sampled value in the table, including
the independently count-refined records.  It should still be viewed as a
finite-data heuristic because conjecture (6) is stated only for fixed \(j\),
not uniformly in the terminal layer.

### (4) Gap-level powers

The proposed law

\[
\gamma_j=N^{j/2+o(1)}
\]

is strongly contradicted, already at \(j=1\).  For the five largest sizes,
\(\gamma_1/N\) is approximately stable while \(\gamma_1/\sqrt N\) grows
rapidly.  The data support the replacement

\[
\boxed{\gamma_j=N^{j+o(1)}\quad\text{for fixed }j.}
\]

Selected scaled values are:

| \(N\) | \(\gamma_1/N\) | \(\gamma_2/N^2\) | \(\gamma_3/N^3\) |
|---:|---:|---:|---:|
| 5000 | 0.0423233 | 2.88393e-4 | 7.06017e-7 |
| 8000 | 0.0442441 | 4.32170e-4 | 8.46057e-7 |
| 12000 | 0.0470703 | 5.02718e-4 | 8.42278e-7 |
| 20000 | 0.0462593 | 6.23176e-4 | 8.66302e-7 |
| 50000 | 0.0461465 | 6.37642e-4 | 7.32535e-7 |

Log-log slopes fitted over \(N\ge5000\) are approximately

\[
1.034,\ 2.328,\ 3.002,\ 3.667,\ 4.237,\ 4.913
\]

for \(j=1,\ldots,6\).  Deeper layers are more affected by finite-size and
terminal-layer effects.  A tentative sharper form for the first few levels is

\[
\gamma_1\sim 0.0462N,\qquad
\gamma_2\sim (6\text{--}7)\times10^{-4}N^2,\qquad
\gamma_3\sim 8\times10^{-7}N^3,
\]

but the constants require more data and better integer-count optimization.

### (5) Central plateau count

The target constant is

\[
1-\frac1\pi=0.681690113816\ldots.
\]

At \(N=20000\), the highest independently count-refined run gives

\[
\frac{n_0}{N}=0.68155,
\]

an absolute error of \(-1.40\times10^{-4}\) and a relative error of about
\(-0.0206\%\).  The \(N=5000\) result is also close, but the \(N=8000\) local
count optimum has \(n_0/N=0.655125\).  This nonmonotone behavior reflects a
very flat integer-count landscape rather than a comparable change in the
spectral radius.

At \(N=8000\), reoptimizing the levels with the nearly conjectured count list

\[
[5455,867,276,88,28,9,3,1]
\]

gives \(\rho=3.141603550799608\), only
\(5.78\times10^{-6}\) below the locally count-refined profile.  Both profiles
have directly checked Rayleigh quotients above \(\pi\).  Therefore conjecture
(5) is plausible and well matched at \(N=20000\), but the present computations
do not identify the count constant as decisively as they identify the gap
exponent or the fact that the limiting value exceeds \(\pi\).

### (6) Fixed noncentral plateau counts

At \(N=20000\), compare the observed fractions with

\[
\frac12\left(1-\frac1\pi\right)\pi^{-j}.
\]

| \(j\) | observed / proposed fraction |
|---:|---:|
| 1 | 0.97793 |
| 2 | 1.06993 |
| 3 | 1.00521 |
| 4 | 0.98597 |
| 5 | 0.98761 |
| 6 | 0.98721 |

This is good agreement for a local integer search, except for a compensating
redistribution between layers 1 and 2.  Successive ratios are generally near
\(\pi\), but outer layers with counts of only a few integers fluctuate
substantially.  Conjecture (6) is supported, with the qualification that the
objective is exceptionally insensitive to nearby count redistributions and
that the \(N=12000\) and \(N=50000\) count lists are not independent tests.

## Prediction for the limiting spectral radius

The candidate sequence is increasing and already gives the finite-dimensional
lower bound

\[
\sup_\lambda\rho(B(\lambda))\big|_{N=50000}
\ge 3.142844355775985.
\]

A nonlinear least-squares fit on \(N\ge1700\) to

\[
\rho_N=C-aN^{-p}
\]

gives

\[
C=3.1432048339,\qquad a=2.36378,\qquad p=0.81233,
\]

with an RMS residual of about \(1.28\times10^{-6}\).  Varying the lower cutoff
from 1700 through 8000 gives fitted limits between approximately
3.1432038 and 3.1432069; restricting to independently count-refined records
from \(N\ge1700\) gives about 3.1432079.

The recommended numerical prediction is therefore

\[
\boxed{\lim_{N\to\infty}\sup_\lambda\rho(B(\lambda))\approx 3.143205,}
\]

with a deliberately wider working interval

\[
\boxed{3.14315\ \text{to}\ 3.14330.}
\]

The central prediction is about

\[
0.00161235
\]

above \(\pi\), or approximately \(0.0513\%\).  The interval is not a
statistical confidence interval; it is an allowance for model choice,
finite-size behavior, the structural profile assumption, and incomplete
global optimization of the integer counts.

Most importantly, the prediction that the limit is greater than \(\pi\) does
not rest only on extrapolation: explicit \(N=8000\) and \(N=50000\) numerical
Rayleigh witnesses already exceed \(\pi\).

## Reproduction

The main dependencies are Python 3.10+, NumPy, SciPy, Numba, and Matplotlib.
Run commands from the package directory.

Refine a supplied profile:

```bash
python large_N_extremal_gap_search.py search \
  --profile profile_N8000.json \
  --vector eigenvector_N8000.npy \
  --level-iterations 4 --count-cycles 1 \
  --out refined_N8000.json --vector-out refined_v_N8000.npy
```

Cross-check it with stricter hierarchical settings and a direct matrix-vector
product:

```bash
python large_N_extremal_gap_search.py certify \
  --profile profile_N8000.json \
  --vector eigenvector_N8000.npy \
  --direct --out certificate_N8000.json
```

Independently sum all unordered pairs in extended precision:

```bash
python verify_large_N_witness.py --N 8000 \
  --results large_N_gap_search_results.json \
  --vectors selected_eigenvectors.npz \
  --extended-terms --out witness_N8000.json
```

Regenerate the plots and fit table:

```bash
python analyze_large_N_gap_results.py \
  --input large_N_gap_search_results.json \
  --out-dir large_N_plots
```

## Principal files

- `large_N_extremal_gap_search.py`: documented matrix-free search and
  certification program.
- `verify_large_N_witness.py`: independent direct all-pairs witness checker.
- `analyze_large_N_gap_results.py`: analysis and plot generator.
- `large_N_gap_search_results.json` / `.csv`: compact result table and metadata.
- `large_N_gap_layers.csv`: one row per \((N,j)\), including conjectured and
  observed count fractions and gap scalings.
- `large_N_gap_arrays.npz`: expanded minimum-normalized gap lists.
- `selected_eigenvectors.npz`: numerical witness vectors for \(N=8000\) and
  \(N=50000\).
- `profile_N8000.json`, `profile_N50000.json`, and matching `.npy` vectors:
  standalone inputs for certification commands.
- `profile_N8000_geometric_counts.json` and its matching vector: the alternate
  near-conjectured count branch that also gives a direct Rayleigh quotient above \(\pi\).
- `pairwise_witness_validation.json`: independent all-pairs checks.
- `rayleigh_certificates.json`: hierarchical-rank and Taylor-error checks.
- `numerical_validation.csv` / `.json`: dense/direct cross-validation summary.
- `limit_fit_models.csv` / `.json`: baseline fit models and recommended limit.
- `limit_fit_sensitivity.csv` / `.json`: cutoff and count-independence sensitivity checks.
- `count_branch_comparison_N8000.json`: quantitative illustration of the flat
  integer-count landscape.
- `requirements_large_N.txt`: Python package requirements.
- `large_N_plots/`: standalone PNG figures.
