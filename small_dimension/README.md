# Small-dimensional unrestricted gap search

This directory contains an unrestricted numerical search over all $N-1$
positive gaps for $3\le N\le48$. No reversal symmetry, inward monotonicity, or
prescribed plateau structure is imposed.

The objective is the spectral radius of the real skew-symmetric matrix

$$
B_{jk}=
\begin{cases}
\displaystyle\frac{\sqrt{\delta_j\delta_k}}{\lambda_j-\lambda_k},&j\ne k,\\[5pt]
0,&j=k,
\end{cases}
$$

where $\delta_j$ is the distance from $\lambda_j$ to its nearest neighboring
point.

This directory including this document was produced by ChatGPT for the purpose of numerical exploration. Not independently audited at this time. - BR

## Parameterization and normalization

The optimizer uses the internal scale convention

```text
sum(gaps) = N - 1
```

because the objective is invariant under a common positive rescaling of all
gaps. Before a result is written, the program rescales the candidate by

```text
gaps_output = gaps_search / min(gaps_search)
```

so every stored gap vector satisfies $\min_j g_j=1$. The corresponding points
use $\lambda_1=1$ and cumulative output gaps. Consequently $\lambda_N$ is
generally not equal to $N$.

## Numerical method

The problem is nonconvex and piecewise smooth. The search combines:

1. deterministic symmetric and asymmetric starting configurations;
2. reproducible random starts in the full log-gap space;
3. L-BFGS-B continuation using a smooth homogeneous approximation to the
   nearest-neighbor minimum;
4. exact-objective local polishing;
5. data-driven plateau refinement after near-equalities have appeared;
6. unrestricted coordinate and random-direction perturbation checks.

The plateau refinement does not impose symmetry or a prescribed gap pattern.
It only merges adjacent coordinates that the unrestricted search has already
made nearly equal.

## Stored run

The committed output was produced with

```bash
python small_dimension/search_full_gaps.py \
  --n-min 3 --n-max 48 \
  --random-starts 12 \
  --seed 20260810 \
  --csv small_dimension/results/candidates_N3_N48.csv \
  --json small_dimension/results/candidates_N3_N48.json
```

The run recorded 589,278 objective evaluations and approximately 365 seconds
of aggregate search time. The exact Python and package versions used for this
original run were not recorded; future JSON output from the script includes
that environment metadata.

## Representative configurations

Here $g^{[r]}$ means that the gap $g$ occurs $r$ consecutive times.

| $N$ | Candidate $\rho(B)$ | Approximate gap pattern | Final $\lambda_N$ |
|---:|---:|---|---:|
| 20 | 2.773086362018684 | $1^{[19]}$ | 20.000000000 |
| 21 | 2.788164034243790 | $3.471534709,1^{[18]},3.471534709$ | 25.943069418 |
| 35 | 2.918555091204003 | $5.834064188^{[2]},1^{[30]},5.834064188^{[2]}$ | 54.336256750 |
| 44 | 2.961460249171716 | $16.88129293,4.912153356^{[2]},1^{[37]},4.912153356^{[2]},16.88129293$ | 91.411199287 |
| 48 | 2.975825380668484 | $19.14517911,5.708841228^{[2]},1^{[41]},5.708841228^{[2]},19.14517911$ | 103.125723136 |

The observed regimes are:

- $3\le N\le20$: every gap is $1$;
- $21\le N\le34$: one enlarged gap at each endpoint;
- $35\le N\le43$: two equal enlarged gaps at each endpoint;
- $44\le N\le48$: one outer level and a two-gap shoulder at each endpoint.

Every nonuniform solution in the stored run has a central minimum-gap plateau.
The candidates are also nearly reversal-symmetric and inward-nonincreasing,
although neither property was imposed.

## Validation and limitations

The script contains derivative, scale-invariance, reversal-invariance, and
normalization self-tests:

```bash
python small_dimension/search_full_gaps.py --self-test
```

The stored run also performed local perturbation checks. These checks increase
confidence that the reported points are good local optima, but they do not prove
global optimality.

## Files

- [`search_full_gaps.py`](search_full_gaps.py): unrestricted search program.
- [`results/candidates_N3_N48.csv`](results/candidates_N3_N48.csv): flattened
  diagnostics, gaps, and point coordinates.
- [`results/candidates_N3_N48.json`](results/candidates_N3_N48.json): nested
  full-precision arrays and search configuration.
