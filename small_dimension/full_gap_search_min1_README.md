# Unrestricted full-gap search with `min(gap) = 1` output

## What changed

`maximize_B_spectral_radius_full_gap_min1.py` performs the same unrestricted,
full-dimensional search as the preceding program: all `N-1` positive gaps are
independent, with no imposed reversal symmetry, inward monotonicity, or limit
on the number or location of gap levels.

The optimizer still uses the numerically convenient internal gauge

```text
sum(gaps) = N - 1
```

because the spectral-radius objective is invariant under a common positive
rescaling of every gap. Before a result is written, the program now applies

```text
gaps_output = gaps_search / min(gaps_search)
```

so every reported gap vector satisfies `min(gaps_output) = 1`. The program
re-evaluates the exact matrix after this rescaling and records the resulting
floating-point discrepancy in `rescaling_spectral_radius_error`.

Lambda values retain `lambda_1 = 1` and are obtained by cumulative summation of
the minimum-normalized gaps. Therefore `lambda_N` is generally larger than
`N`; it is no longer forced to equal `N`.

## Reproducible run used for the attached output

```bash
python maximize_B_spectral_radius_full_gap_min1.py \
  --n-min 3 --n-max 48 \
  --random-starts 12 \
  --seed 20260810 \
  --csv full_gap_search_min1_results_N3_N48.csv \
  --json full_gap_search_min1_results_N3_N48.json
```

The completed run recorded **589,278 objective evaluations** and
**364.579 aggregate search seconds** across `N=3,...,48`.

## Validation

- All 46 rows have an output minimum gap equal to exactly `1.0` in the stored
  floating-point arrays; the maximum recorded `|min(gaps)-1|` is
  `0.000e+00`.
- The largest spectral-radius discrepancy caused by evaluating the two common
  scales is `1.021e-14`.
- The largest spectral-radius difference from the preceding unrestricted run
  is `1.021e-14`.
- The largest absolute difference from the supplied structured run is
  `5.329e-15`.
- Every nonuniform solution has its minimum plateau spanning the central gap
  position or positions. Every reported solution is inward-nonincreasing,
  although this property was not imposed.
- The maximum reversal discrepancy is `4.204e-10` relative to the mean
  gap. The local perturbation checks found no improving tested direction.

## Representative minimum-normalized configurations

Here `a^k` denotes `k` consecutive gaps equal to `a`.

| N | Spectral radius | Gap pattern | Final lambda | Minimum-gap positions (1-based) |
|---:|---:|---|---:|---:|
| 20 | 2.773086362018684 | `1^19` | 20.000000000 | 1–19 |
| 21 | 2.788164034243790 | `3.471534709^1 | 1^18 | 3.471534709^1` | 25.943069418 | 2–19 |
| 35 | 2.918555091204003 | `5.834064188^2 | 1^30 | 5.834064188^2` | 54.336256750 | 3–32 |
| 44 | 2.961460249171716 | `16.88129293^1 | 4.912153356^2 | 1^37 | 4.912153356^2 | 16.88129293^1` | 91.411199287 | 4–40 |
| 48 | 2.975825380668484 | `19.14517911^1 | 5.708841228^2 | 1^41 | 5.708841228^2 | 19.14517911^1` | 103.125723136 | 4–44 |

The qualitative regimes are unchanged by normalization:

- `N=3,...,20`: every gap is `1`.
- `N=21,...,34`: one enlarged gap occurs at each endpoint, with a central unit
  plateau.
- `N=35,...,43`: two equal enlarged gaps occur at each endpoint, with a central
  unit plateau.
- `N=44,...,48`: one outer level and a two-gap shoulder occur at each endpoint,
  followed by a central unit plateau.

## Output columns added for the new normalization

The CSV and JSON now explicitly include:

- `output_normalization`: `min(gap)=1`;
- `search_gauge_min_gap`: the minimum before output rescaling;
- `output_scale_from_search_gauge`: the factor applied to every gap;
- `lambda_span` and `lambda_final`;
- `rescaling_spectral_radius_error`;
- the count and first/last 1-based positions of minimum-sized gaps.

The `gap_j` and `lambda_j` columns themselves contain only the new
minimum-normalized values.

## Important limitation

This remains a multistart numerical search over a nonconvex, piecewise-smooth
objective, not a proof of global optimality. The change made here is solely a
scale convention for reported configurations; it does not add or remove any
candidate gap shapes.

## Files

- `maximize_B_spectral_radius_full_gap_min1.py`: documented search program.
- `full_gap_search_min1_results_N3_N48.csv`: diagnostics, all gaps, and all
  lambda values under `min(gap)=1` normalization.
- `full_gap_search_min1_results_N3_N48.json`: full-precision arrays and run
  configuration.
- `full_gap_search_min1_run_N3_N48.txt`: completed terminal output plus
  post-run validation.
- `full_gap_min1_vs_structured_comparison.csv`: row-by-row comparison with the
  preceding unrestricted output and the supplied structured search.
