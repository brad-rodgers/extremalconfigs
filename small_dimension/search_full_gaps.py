#!/usr/bin/env python3
"""Unrestricted numerical search for the B(lambda) spectral-radius problem.

This program maximizes the spectral radius of the real skew-symmetric matrix

    B[m,n] = sqrt(delta[m] * delta[n]) / (lambda[m] - lambda[n]),  m != n,
    B[m,m] = 0,

where lambda[0] < ... < lambda[N-1], the consecutive gaps are

    g[j] = lambda[j+1] - lambda[j] > 0,

and

    delta[0]   = g[0],
    delta[N-1] = g[N-2],
    delta[k]   = min(g[k-1], g[k])       for 1 <= k <= N-2.

This implementation imposes none of the following conditions on the gaps:

* reversal symmetry;
* monotonicity from either endpoint toward the centre;
* a prescribed number or location of gap levels ("jumps").

Every one of the M=N-1 gaps is an independent optimization coordinate.  The
optimizer removes the irrelevant common scale by using the internal numerical
gauge sum(g)=N-1.  Additive log-ratio variables span the entire open simplex of
strictly positive gap sequences in that gauge:

    g_search[j] = (N-1) * exp(y[j]) / sum_k exp(y[k]),    y[N-2] = 0.

The reported configuration is then rescaled without changing its shape:

    g_output[j] = g_search[j] / min_k g_search[k].

Consequently every CSV/JSON output gap vector has min(g_output)=1.  Lambda
values use lambda[0]=1 and cumulative output gaps, so lambda[N-1] is generally
not N under this reporting normalization.  This final rescaling cannot change
B or its spectral radius because the objective is invariant under a common
positive scaling of every gap.

The coordinate map itself spans the entire open simplex.  For floating-point
safety, a finite run places a configurable guard on the log-ratios
(``--logit-bound``, default 10).  The default permits relative gap ratios up to
exp(20), far beyond the largest ratio found here (about 19.14), and the guard
was inactive at every reported solution.  This is a numerical range guard, not
a symmetry, monotonicity, or gap-pattern assumption.

The problem is nonconvex and piecewise smooth because of the interior minimum
in delta.  A finite computation cannot literally enumerate the uncountable
set of positive real gap sequences, and this code does not claim a proof of
global optimality.  It is an unrestricted full-dimensional numerical search.
To make that search robust, it combines:

1. deterministic starts with varied symmetric and asymmetric shapes;
2. reproducible random starts in the full log-gap space;
3. L-BFGS-B continuation using a smooth, homogeneous approximation to min;
4. exact-objective local polishing using a Clarke-style tie subgradient;
5. data-driven plateau refinement after near-equalities have been discovered;
6. unrestricted coordinate and random-direction perturbation checks.

The plateau refinement does not prescribe a pattern.  It only merges adjacent
coordinates that the unrestricted smooth search has already made nearly equal,
then optimizes every discovered run independently.  Left and right runs are
not tied together, so reversal symmetry is never imposed.

Example
-------
Run the default N=3,...,48 search and write minimum-gap-normalized CSV and
JSON output:

    python small_dimension/search_full_gaps.py

Use more random starts and local validation restarts:

    python small_dimension/search_full_gaps.py --thorough

Run numerical derivative, invariance, and output-normalization tests only:

    python small_dimension/search_full_gaps.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """Numerical controls shared by all values of N."""

    random_starts: int = 16
    seed: int = 20260810
    smooth_beta: float = 256.0
    logit_bound: float = 10.0
    smooth_maxiter: int = 140
    exact_maxiter: int = 350
    shortlist: int = 8
    merge_tolerances: tuple[float, ...] = (0.008, 0.02, 0.05)
    validation_restarts: int = 4
    random_validation_directions: int = 48
    thorough: bool = False


@dataclass
class Candidate:
    """One exact-objective candidate produced during the search."""

    rho: float
    gaps: np.ndarray
    source: str


@dataclass
class SearchResult:
    """Best result and diagnostics for one matrix size N."""

    N: int
    rho: float
    rho_uniform: float
    improvement_over_uniform: float
    elapsed_seconds: float
    objective_evaluations: int
    start_count: int
    candidate_count: int
    best_source: str
    gaps: np.ndarray
    lambdas: np.ndarray
    search_gauge_min_gap: float
    output_scale_from_search_gauge: float
    lambda_span: float
    lambda_final: float
    rescaling_spectral_radius_error: float
    min_gap: float
    max_gap: float
    gap_ratio: float
    min_gap_count: int
    min_gap_first_1_based: int
    min_gap_last_1_based: int
    gap_run_count: int
    gap_pattern: str
    reversal_error: float
    inward_nonincreasing: bool
    max_coordinate_perturbation_gain: float
    max_random_perturbation_gain: float


# ---------------------------------------------------------------------------
# Mathematical objective and analytic derivatives
# ---------------------------------------------------------------------------


class FullGapProblem:
    """Objective and derivatives on the complete positive gap simplex.

    Parameters
    ----------
    N:
        Number of lambda points.  There are M=N-1 independent positive gaps.

    Notes
    -----
    ``value_and_gradient_gaps`` differentiates the largest eigenvalue of iB,
    which equals the spectral radius of the real skew-symmetric B.  The
    eigenvector derivative is valid when the top eigenvalue is simple, as it is
    at all solutions encountered in this search.
    """

    def __init__(self, N: int):
        if N < 3:
            raise ValueError("N must be at least 3")
        self.N = int(N)
        self.M = self.N - 1
        self.total_gap = float(self.M)
        self.evaluations = 0

    def logits_to_gaps(self, u: np.ndarray) -> np.ndarray:
        """Map M-1 additive log-ratios to M positive gaps summing to M."""
        u = np.asarray(u, dtype=float)
        if u.shape != (self.M - 1,):
            raise ValueError(f"expected shape {(self.M - 1,)}, got {u.shape}")
        y = np.concatenate((u, np.array([0.0])))
        y -= np.max(y)
        weights = np.exp(y)
        return self.total_gap * weights / weights.sum()

    def gaps_to_logits(self, gaps: np.ndarray) -> np.ndarray:
        """Inverse log-ratio coordinates, using the final gap as the gauge."""
        gaps = np.asarray(gaps, dtype=float)
        if gaps.shape != (self.M,) or np.any(gaps <= 0.0):
            raise ValueError(f"gaps must be positive with shape {(self.M,)}")
        return np.log(gaps[:-1] / gaps[-1])

    @staticmethod
    def normalize_gaps(gaps: np.ndarray, total: float) -> np.ndarray:
        """Return a positive copy scaled to a fixed total search gauge."""
        gaps = np.asarray(gaps, dtype=float).copy()
        if gaps.ndim != 1 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
            raise ValueError("all gaps must be finite and strictly positive")
        return total * gaps / gaps.sum()

    @staticmethod
    def normalize_gaps_to_minimum_one(gaps: np.ndarray) -> np.ndarray:
        """Return a positive copy rescaled so its smallest coordinate is one.

        This is the reporting normalization requested for the output files.
        It changes neither gap ratios nor the spectral radius of B.
        """
        gaps = np.asarray(gaps, dtype=float).copy()
        if gaps.ndim != 1 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
            raise ValueError("all gaps must be finite and strictly positive")
        return gaps / float(np.min(gaps))

    def _delta_and_jacobian(
        self,
        gaps: np.ndarray,
        beta: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return delta and d(delta)/d(gaps).

        When ``beta is None``, use the exact minimum.  At an exact numerical
        tie, split the derivative equally between the two adjacent gaps; this
        is one element of the Clarke subdifferential.

        For finite beta, use the homogeneous power soft-min

            s_beta(a,b) = ((a**(-beta) + b**(-beta))/2)**(-1/beta).

        It is positive, scale equivariant, exactly equals a when a=b, and
        converges monotonically to min(a,b) as beta tends to infinity.
        """
        N, M = self.N, self.M
        delta = np.empty(N, dtype=float)
        jac = np.zeros((N, M), dtype=float)

        delta[0] = gaps[0]
        jac[0, 0] = 1.0
        delta[-1] = gaps[-1]
        jac[-1, -1] = 1.0

        left = gaps[:-1]
        right = gaps[1:]
        rows = np.arange(1, N - 1)
        left_cols = np.arange(M - 1)
        right_cols = np.arange(1, M)

        if beta is None:
            delta[1:-1] = np.minimum(left, right)
            # Only machine-level ties are split here.  Wider near-equality
            # handling is done explicitly by the plateau-refinement stage.
            tie_scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
            tied = np.abs(left - right) <= 64.0 * np.finfo(float).eps * tie_scale
            take_left = (left < right) & ~tied
            take_right = (right < left) & ~tied
            jac[rows[take_left], left_cols[take_left]] = 1.0
            jac[rows[take_right], right_cols[take_right]] = 1.0
            jac[rows[tied], left_cols[tied]] = 0.5
            jac[rows[tied], right_cols[tied]] = 0.5
        else:
            if beta <= 0.0:
                raise ValueError("smooth beta must be positive")
            log_left = np.log(left)
            log_right = np.log(right)
            stacked = np.stack((-beta * log_left, -beta * log_right), axis=0)
            normalizer = logsumexp(stacked, axis=0)
            log_delta = -(normalizer - math.log(2.0)) / beta
            interior = np.exp(log_delta)
            delta[1:-1] = interior

            # Softmax weights are d(log delta)/d(log left/right).
            weights = np.exp(stacked - normalizer)
            jac[rows, left_cols] = interior * weights[0] / left
            jac[rows, right_cols] = interior * weights[1] / right

        return delta, jac

    def value_and_gradient_gaps(
        self,
        gaps: np.ndarray,
        beta: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return spectral radius and derivative with respect to all gaps."""
        self.evaluations += 1
        gaps = np.asarray(gaps, dtype=float)
        if gaps.shape != (self.M,) or np.any(gaps <= 0.0):
            raise ValueError("invalid gap vector")

        delta, d_delta_d_gaps = self._delta_and_jacobian(gaps, beta)
        x = np.concatenate((np.array([0.0]), np.cumsum(gaps)))
        denominator = x[:, None] - x[None, :]
        np.fill_diagonal(denominator, 1.0)
        if np.any(denominator[~np.eye(self.N, dtype=bool)] == 0.0):
            raise FloatingPointError("two lambda values coalesced numerically")

        root_delta = np.sqrt(delta)
        B = (root_delta[:, None] * root_delta[None, :]) / denominator
        np.fill_diagonal(B, 0.0)

        values, vectors = eigh(
            1j * B,
            subset_by_index=[self.N - 1, self.N - 1],
            check_finite=False,
        )
        rho = float(values[0])
        eigenvector = vectors[:, 0]

        # Real sensitivity of rho with respect to each B[m,n].
        sensitivity_B = np.real(
            1j * np.conj(eigenvector)[:, None] * eigenvector[None, :]
        )
        product = sensitivity_B * B
        d_rho_d_delta = product.sum(axis=1) / delta
        d_rho_d_x = -2.0 * (product / denominator).sum(axis=1)

        # x[k] = sum_{j<k} gaps[j], so gap[j] affects x[j+1],...,x[N-1].
        d_rho_d_gaps_from_x = np.cumsum(d_rho_d_x[:0:-1])[::-1]
        d_rho_d_gaps = (
            d_rho_d_gaps_from_x + d_delta_d_gaps.T @ d_rho_d_delta
        )
        return rho, d_rho_d_gaps

    def value_and_gradient_logits(
        self,
        u: np.ndarray,
        beta: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return spectral radius and derivative in additive log-ratios."""
        gaps = self.logits_to_gaps(u)
        rho, gradient_gaps = self.value_and_gradient_gaps(gaps, beta)

        probabilities = gaps / self.total_gap
        weighted_mean = float(np.dot(probabilities, gradient_gaps))
        gradient_full_logits = gaps * (gradient_gaps - weighted_mean)
        return rho, gradient_full_logits[:-1]

    def exact_value(self, gaps: np.ndarray) -> float:
        """Evaluate the exact nonsmoothed objective."""
        return self.value_and_gradient_gaps(gaps, beta=None)[0]


# ---------------------------------------------------------------------------
# Data-driven equality-run refinement
# ---------------------------------------------------------------------------


class GroupGapProblem:
    """Optimize independently valued contiguous gap runs.

    ``labels[j]`` identifies the discovered run containing gap j.  Repeated
    numerical values in nonadjacent runs remain independent.  In particular,
    left and right sides are never coupled by this class.
    """

    def __init__(self, base: FullGapProblem, labels: np.ndarray):
        labels = np.asarray(labels, dtype=int)
        if labels.shape != (base.M,):
            raise ValueError("invalid run-label shape")
        if labels[0] != 0 or np.any(np.diff(labels) < 0):
            raise ValueError("run labels must be consecutive and nondecreasing")
        if not np.array_equal(np.unique(labels), np.arange(labels[-1] + 1)):
            raise ValueError("run labels must be 0,1,...,Q-1")
        self.base = base
        self.labels = labels
        self.Q = int(labels[-1]) + 1
        self.counts = np.bincount(labels, minlength=self.Q).astype(float)

    def logits_to_levels_and_gaps(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map Q-1 log-ratios to Q run levels and the expanded gap vector."""
        if self.Q == 1:
            if np.asarray(u).size:
                raise ValueError("one run has no free log-ratio")
            levels = np.array([self.base.total_gap / self.counts[0]])
            return levels, levels[self.labels]

        u = np.asarray(u, dtype=float)
        if u.shape != (self.Q - 1,):
            raise ValueError(f"expected shape {(self.Q - 1,)}, got {u.shape}")
        y = np.concatenate((u, np.array([0.0])))
        y -= np.max(y)
        weights = np.exp(y)
        levels = self.base.total_gap * weights / np.dot(self.counts, weights)
        return levels, levels[self.labels]

    def gaps_to_logits(self, gaps: np.ndarray) -> np.ndarray:
        """Construct group log-ratios from the mean value in each run."""
        if self.Q == 1:
            return np.empty(0, dtype=float)
        levels = np.array(
            [np.mean(gaps[self.labels == q]) for q in range(self.Q)], dtype=float
        )
        return np.log(levels[:-1] / levels[-1])

    def value_and_gradient(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the exact objective and derivative in run-level coordinates."""
        levels, gaps = self.logits_to_levels_and_gaps(u)
        rho, gradient_gaps = self.base.value_and_gradient_gaps(gaps, beta=None)
        gradient_levels = np.bincount(
            self.labels, weights=gradient_gaps, minlength=self.Q
        )
        scale_term = float(np.dot(levels, gradient_levels))
        gradient_logits = (
            levels * gradient_levels
            - (self.counts * levels / self.base.total_gap) * scale_term
        )
        return rho, gradient_logits[:-1]


def labels_from_near_equal_gaps(gaps: np.ndarray, relative_tolerance: float) -> np.ndarray:
    """Label maximal contiguous runs whose adjacent values are nearly equal."""
    gaps = np.asarray(gaps, dtype=float)
    labels = np.zeros(gaps.size, dtype=int)
    current = 0
    for j in range(1, gaps.size):
        scale = max(abs(gaps[j - 1]), abs(gaps[j]), 1e-15)
        if abs(gaps[j] - gaps[j - 1]) > relative_tolerance * scale:
            current += 1
        labels[j] = current
    return labels


# ---------------------------------------------------------------------------
# Starts, local searches, and diagnostics
# ---------------------------------------------------------------------------


def _scaled_positive(values: np.ndarray, total: float) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 1e-14)
    return total * values / values.sum()


def interpolate_warm_start(previous_gaps: np.ndarray, new_length: int) -> np.ndarray:
    """Interpolate a previous solution in log-gap space to a new dimension."""
    old = np.asarray(previous_gaps, dtype=float)
    old_grid = np.linspace(0.0, 1.0, old.size)
    new_grid = np.linspace(0.0, 1.0, new_length)
    return np.exp(np.interp(new_grid, old_grid, np.log(old)))


def generate_starts(
    problem: FullGapProblem,
    config: SearchConfig,
    rng: np.random.Generator,
    previous_gaps: np.ndarray | None,
) -> list[tuple[str, np.ndarray]]:
    """Generate varied full-dimensional starts without constraining any run.

    The deterministic starts are deliberately heterogeneous.  They improve
    coverage but are not retained as constraints: every start is subsequently
    optimized in all M-1 log-ratio directions.
    """
    M = problem.M
    starts: list[tuple[str, np.ndarray]] = []

    def add(name: str, values: np.ndarray) -> None:
        gaps = _scaled_positive(values, problem.total_gap)
        u = problem.gaps_to_logits(gaps)
        u = np.clip(u, -config.logit_bound, config.logit_bound)
        starts.append((name, u))

    add("uniform", np.ones(M))

    if previous_gaps is not None:
        add("continuation", interpolate_warm_start(previous_gaps, M))

    # One-level edge blocks at several widths and amplitudes.
    for width in range(1, min(5, M // 2) + 1):
        for ratio in (4.0, 15.0):
            values = np.ones(M)
            values[:width] = ratio
            values[-width:] = ratio
            add(f"edge_block_w{width}_r{ratio:g}", values)

    # Two-level edge profiles.  These are useful starts near sharp boundary
    # layers, but the ensuing optimizer can move every individual coordinate.
    if M >= 7:
        for shoulder_width in (1, 2, 3):
            for outer, shoulder in ((12.0, 4.0), (18.0, 6.0)):
                values = np.ones(M)
                values[0] = values[-1] = outer
                values[1 : 1 + shoulder_width] = shoulder
                values[-1 - shoulder_width : -1] = shoulder
                add(
                    f"two_level_w{shoulder_width}_o{outer:g}_s{shoulder:g}",
                    values,
                )

    # Asymmetric and oscillatory deterministic starts explore directions that
    # are excluded by a reversal-symmetric parameterization.
    grid = np.linspace(-1.0, 1.0, M)
    add("left_to_right_ramp", np.exp(1.6 * grid))
    add("right_to_left_ramp", np.exp(-1.6 * grid))
    add("cosine_mode", np.exp(1.2 * np.cos(np.pi * grid)))
    add("sine_mode", np.exp(1.2 * np.sin(np.pi * grid)))
    add("alternating", np.exp(0.9 * ((-1.0) ** np.arange(M))))

    # Localized interior perturbations at several positions.
    if M >= 5:
        index = np.arange(M)
        for fraction in (0.25, 0.5, 0.75):
            centre = fraction * (M - 1)
            width = max(1.0, 0.08 * M)
            bump = np.exp(-0.5 * ((index - centre) / width) ** 2)
            add(f"interior_bump_{fraction:.2f}", np.exp(1.8 * bump))

    # Random starts alternate logistic-normal and Dirichlet distributions over
    # the complete simplex.  No sorting or symmetrization is performed.
    log_scales = (0.25, 0.6, 1.2, 2.2, 3.5)
    dirichlet_alphas = (0.35, 0.8, 2.0)
    for k in range(config.random_starts):
        if k % 2 == 0:
            sigma = log_scales[(k // 2) % len(log_scales)]
            logits = np.clip(rng.normal(0.0, sigma, size=M), -9.0, 9.0)
            add(f"random_logistic_{k:03d}", np.exp(logits - logits.max()))
        else:
            alpha = dirichlet_alphas[(k // 2) % len(dirichlet_alphas)]
            add(f"random_dirichlet_{k:03d}", rng.dirichlet(np.full(M, alpha)))

    # Deduplicate starts after normalization and gauge fixing.
    unique: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, u in starts:
        key = tuple(np.round(u, 11))
        if key not in seen:
            seen.add(key)
            unique.append((name, u))
    return unique


def optimize_logits(
    problem: FullGapProblem,
    start_u: np.ndarray,
    config: SearchConfig,
    beta: float | None,
    maxiter: int,
) -> tuple[np.ndarray, str]:
    """Run one bounded L-BFGS-B solve in full log-gap coordinates."""

    def objective(u: np.ndarray) -> tuple[float, np.ndarray]:
        rho, gradient = problem.value_and_gradient_logits(u, beta=beta)
        return -rho, -gradient

    bounds = [(-config.logit_bound, config.logit_bound)] * (problem.M - 1)
    result = minimize(
        objective,
        np.clip(start_u, -config.logit_bound, config.logit_bound),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "ftol": 1e-10 if beta is not None else 2e-14,
            "gtol": 2e-6 if beta is not None else 2e-9,
            "maxiter": maxiter,
            "maxfun": max(1200, 12 * maxiter),
            "maxls": 60,
        },
    )
    return np.asarray(result.x, dtype=float), str(result.message)


def refine_discovered_runs(
    problem: FullGapProblem,
    gaps: np.ndarray,
    relative_tolerance: float,
    config: SearchConfig,
) -> Candidate:
    """Optimize all data-discovered contiguous equality runs independently."""
    labels = labels_from_near_equal_gaps(gaps, relative_tolerance)
    grouped = GroupGapProblem(problem, labels)

    if grouped.Q == 1:
        _, refined_gaps = grouped.logits_to_levels_and_gaps(np.empty(0))
        return Candidate(problem.exact_value(refined_gaps), refined_gaps, "run_refine_Q1")

    start_u = grouped.gaps_to_logits(gaps)

    def objective(u: np.ndarray) -> tuple[float, np.ndarray]:
        rho, gradient = grouped.value_and_gradient(u)
        return -rho, -gradient

    bounds = [(-config.logit_bound, config.logit_bound)] * (grouped.Q - 1)
    result = minimize(
        objective,
        np.clip(start_u, -config.logit_bound, config.logit_bound),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "ftol": 2e-14,
            "gtol": 2e-10,
            "maxiter": config.exact_maxiter,
            "maxfun": max(1800, 12 * config.exact_maxiter),
            "maxls": 80,
        },
    )
    refined_u = np.asarray(result.x, dtype=float)
    _, refined_gaps = grouped.logits_to_levels_and_gaps(refined_u)
    rho = problem.exact_value(refined_gaps)
    source = f"run_refine_tol{relative_tolerance:g}_Q{grouped.Q}"

    # On a discovered equality pattern the exact objective is smooth in the
    # run levels, so a small root solve can sharpen flat L-BFGS-B directions.
    # Limit this step to low-dimensional discovered patterns; a genuinely
    # all-distinct candidate remains handled by the unrestricted exact solve.
    if grouped.Q <= 12:
        def stationarity(u: np.ndarray) -> np.ndarray:
            return grouped.value_and_gradient(u)[1]

        before_norm = float(np.linalg.norm(stationarity(refined_u), ord=np.inf))
        stationary = least_squares(
            stationarity,
            refined_u,
            bounds=(-config.logit_bound, config.logit_bound),
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=500,
        )
        root_u = np.asarray(stationary.x, dtype=float)
        if np.all(np.isfinite(root_u)):
            after_norm = float(np.linalg.norm(stationary.fun, ord=np.inf))
            _, root_gaps = grouped.logits_to_levels_and_gaps(root_u)
            root_rho = problem.exact_value(root_gaps)
            if (
                root_rho >= rho - 5e-13
                and after_norm <= max(1e-9, 0.25 * before_norm)
            ):
                refined_u = root_u
                refined_gaps = root_gaps
                rho = root_rho
                source += ":stationary"

    return Candidate(rho=rho, gaps=refined_gaps, source=source)


def deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep the best source for numerically identical normalized gap vectors."""
    best_by_key: dict[tuple[float, ...], Candidate] = {}
    for candidate in candidates:
        key = tuple(np.round(candidate.gaps, 10))
        old = best_by_key.get(key)
        if old is None or candidate.rho > old.rho:
            best_by_key[key] = candidate
    return sorted(best_by_key.values(), key=lambda c: c.rho, reverse=True)


def gap_runs(gaps: np.ndarray, tolerance: float = 2e-6) -> list[tuple[float, int]]:
    """Compress consecutive numerically equal gaps for human-readable output."""
    runs: list[tuple[float, int]] = []
    for value in np.asarray(gaps, dtype=float):
        if runs:
            previous, count = runs[-1]
            scale = max(abs(previous), abs(value), 1e-15)
            if abs(value - previous) <= tolerance * scale:
                # Average to make the displayed run value less side-dependent.
                runs[-1] = ((previous * count + float(value)) / (count + 1), count + 1)
                continue
        runs.append((float(value), 1))
    return runs


def compressed_gap_pattern(gaps: np.ndarray) -> str:
    return " | ".join(f"{value:.10g}^{count}" for value, count in gap_runs(gaps))


def reversal_error(gaps: np.ndarray) -> float:
    """Maximum reversal discrepancy divided by the mean gap.

    Dividing by the mean makes this diagnostic independent of the chosen
    common-scale normalization.
    """
    return float(np.max(np.abs(gaps - gaps[::-1])) / np.mean(gaps))


def minimum_gap_indices(
    gaps: np.ndarray,
    relative_tolerance: float = 2e-6,
) -> np.ndarray:
    """Return zero-based indices numerically belonging to the minimum plateau."""
    gaps = np.asarray(gaps, dtype=float)
    minimum = float(np.min(gaps))
    return np.flatnonzero(np.abs(gaps - minimum) <= relative_tolerance * minimum)


def is_inward_nonincreasing(gaps: np.ndarray, tolerance: float = 2e-6) -> bool:
    """Check the empirical endpoint-to-centre monotonicity after the search."""
    M = gaps.size
    left = gaps[: (M + 1) // 2]
    right = gaps[::-1][: (M + 1) // 2]
    left_ok = np.all(np.diff(left) <= tolerance * np.maximum(1.0, np.abs(left[:-1])))
    right_ok = np.all(np.diff(right) <= tolerance * np.maximum(1.0, np.abs(right[:-1])))
    return bool(left_ok and right_ok)


def perturbation_diagnostics(
    problem: FullGapProblem,
    gaps: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    random_directions: int,
) -> tuple[float, float]:
    """Probe unrestricted nearby directions around the reported solution.

    Positive values would indicate an improving perturbation and therefore a
    failed local-optimality check.  Small negative values are expected at a
    nonsmooth local maximum.
    """
    u = problem.gaps_to_logits(gaps)
    coordinate_gain = -math.inf
    coordinate_step = 1e-5
    for j in range(u.size):
        for sign in (-1.0, 1.0):
            trial = u.copy()
            trial[j] += sign * coordinate_step
            coordinate_gain = max(
                coordinate_gain,
                problem.exact_value(problem.logits_to_gaps(trial)) - rho,
            )

    random_gain = -math.inf
    if random_directions <= 0:
        random_gain = math.nan
    else:
        for k in range(random_directions):
            direction = rng.normal(size=u.size)
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                continue
            direction /= norm
            step = (1e-4, 1e-3, 1e-2)[k % 3]
            for sign in (-1.0, 1.0):
                trial = u + sign * step * direction
                random_gain = max(
                    random_gain,
                    problem.exact_value(problem.logits_to_gaps(trial)) - rho,
                )
    return float(coordinate_gain), float(random_gain)


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------


def search_one_N(
    N: int,
    config: SearchConfig,
    previous_gaps: np.ndarray | None = None,
) -> SearchResult:
    """Run the unrestricted multistart search for one N."""
    start_time = time.perf_counter()
    problem = FullGapProblem(N)
    rng = np.random.default_rng(config.seed + 104729 * N)
    starts = generate_starts(problem, config, rng, previous_gaps)

    candidates: list[Candidate] = []

    # Evaluate and smoothly optimize every start in all M-1 coordinates.
    for name, start_u in starts:
        raw_gaps = problem.logits_to_gaps(start_u)
        candidates.append(Candidate(problem.exact_value(raw_gaps), raw_gaps, f"raw:{name}"))

        smooth_u, _ = optimize_logits(
            problem,
            start_u,
            config,
            beta=config.smooth_beta,
            maxiter=config.smooth_maxiter,
        )
        smooth_gaps = problem.logits_to_gaps(smooth_u)
        candidates.append(
            Candidate(problem.exact_value(smooth_gaps), smooth_gaps, f"smooth:{name}")
        )

    candidates = deduplicate_candidates(candidates)

    # Exact full-dimensional polishing and data-driven run refinement of the
    # best candidates.  Nothing here ties reversal partners together.
    initial_shortlist = candidates[: min(config.shortlist, len(candidates))]
    additions: list[Candidate] = []
    for candidate in initial_shortlist:
        exact_u, _ = optimize_logits(
            problem,
            problem.gaps_to_logits(candidate.gaps),
            config,
            beta=None,
            maxiter=config.exact_maxiter,
        )
        exact_gaps = problem.logits_to_gaps(exact_u)
        additions.append(
            Candidate(problem.exact_value(exact_gaps), exact_gaps, f"exact:{candidate.source}")
        )

        for tolerance in config.merge_tolerances:
            additions.append(
                refine_discovered_runs(problem, candidate.gaps, tolerance, config)
            )

    candidates = deduplicate_candidates([*candidates, *additions])

    # Full-dimensional asymmetric restarts around the current best basin.
    restart_count = config.validation_restarts * (2 if config.thorough else 1)
    for restart in range(restart_count):
        best_u = problem.gaps_to_logits(candidates[0].gaps)
        sigma = (0.03, 0.12, 0.35, 0.8)[restart % 4]
        perturbed_u = np.clip(
            best_u + rng.normal(0.0, sigma, size=best_u.size),
            -config.logit_bound,
            config.logit_bound,
        )
        smooth_u, _ = optimize_logits(
            problem,
            perturbed_u,
            config,
            beta=config.smooth_beta * (4.0 if config.thorough else 1.0),
            maxiter=config.smooth_maxiter * (2 if config.thorough else 1),
        )
        smooth_gaps = problem.logits_to_gaps(smooth_u)
        restart_candidate = Candidate(
            problem.exact_value(smooth_gaps),
            smooth_gaps,
            f"restart_{restart}",
        )
        candidates.append(restart_candidate)
        for tolerance in config.merge_tolerances:
            candidates.append(
                refine_discovered_runs(problem, smooth_gaps, tolerance, config)
            )
        candidates = deduplicate_candidates(candidates)

    best = candidates[0]

    # Keep the smooth optimizer's fixed-sum gauge for numerical diagnostics,
    # then convert only the reported configuration to min(gap)=1.  The exact
    # objective is evaluated in both gauges as an explicit invariance check.
    search_gaps = FullGapProblem.normalize_gaps(best.gaps, problem.total_gap)
    rho_search_gauge = problem.exact_value(search_gaps)
    search_gauge_min_gap = float(np.min(search_gaps))
    gaps = FullGapProblem.normalize_gaps_to_minimum_one(search_gaps)
    output_scale = 1.0 / search_gauge_min_gap
    rho = problem.exact_value(gaps)
    rescaling_error = abs(rho - rho_search_gauge)
    if rescaling_error > 5e-11:
        raise RuntimeError(
            f"scale-invariance check failed for N={N}: {rescaling_error:.3e}"
        )

    # Retain lambda[0]=1 as the origin convention.  Because the smallest gap,
    # rather than the total span, is fixed, the final lambda is generally not N.
    lambdas = 1.0 + np.concatenate((np.array([0.0]), np.cumsum(gaps)))
    lambdas[0] = 1.0

    uniform_gaps = np.ones(problem.M, dtype=float)
    rho_uniform = problem.exact_value(uniform_gaps)
    coordinate_gain, random_gain = perturbation_diagnostics(
        problem,
        search_gaps,
        rho_search_gauge,
        rng,
        config.random_validation_directions * (2 if config.thorough else 1),
    )

    elapsed = time.perf_counter() - start_time
    runs = gap_runs(gaps)
    minimum_indices = minimum_gap_indices(gaps)
    return SearchResult(
        N=N,
        rho=rho,
        rho_uniform=rho_uniform,
        improvement_over_uniform=rho - rho_uniform,
        elapsed_seconds=elapsed,
        objective_evaluations=problem.evaluations,
        start_count=len(starts),
        candidate_count=len(candidates),
        best_source=best.source,
        gaps=gaps,
        lambdas=lambdas,
        search_gauge_min_gap=search_gauge_min_gap,
        output_scale_from_search_gauge=output_scale,
        lambda_span=float(np.sum(gaps)),
        lambda_final=float(lambdas[-1]),
        rescaling_spectral_radius_error=rescaling_error,
        min_gap=float(np.min(gaps)),
        max_gap=float(np.max(gaps)),
        gap_ratio=float(np.max(gaps) / np.min(gaps)),
        min_gap_count=int(minimum_indices.size),
        min_gap_first_1_based=int(minimum_indices[0] + 1),
        min_gap_last_1_based=int(minimum_indices[-1] + 1),
        gap_run_count=len(runs),
        gap_pattern=compressed_gap_pattern(gaps),
        reversal_error=reversal_error(gaps),
        inward_nonincreasing=is_inward_nonincreasing(gaps),
        max_coordinate_perturbation_gain=coordinate_gain,
        max_random_perturbation_gain=random_gain,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path: Path, results: Sequence[SearchResult]) -> None:
    """Write diagnostics plus every min-normalized gap and lambda coordinate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    max_n = max(result.N for result in results)
    scalar_fields = [
        "N",
        "rho",
        "rho_uniform",
        "improvement_over_uniform",
        "search_seconds",
        "objective_evaluations",
        "start_count",
        "candidate_count",
        "best_source",
        "output_normalization",
        "search_gauge_min_gap",
        "output_scale_from_search_gauge",
        "lambda_span",
        "lambda_final",
        "rescaling_spectral_radius_error",
        "min_gap",
        "max_gap",
        "gap_ratio",
        "min_gap_count",
        "min_gap_first_1_based",
        "min_gap_last_1_based",
        "gap_run_count",
        "gap_pattern",
        "reversal_error",
        "inward_nonincreasing",
        "max_coordinate_perturbation_gain",
        "max_random_perturbation_gain",
    ]
    fields = (
        scalar_fields
        + [f"gap_{j}" for j in range(1, max_n)]
        + [f"lambda_{j}" for j in range(1, max_n + 1)]
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row: dict[str, object] = {
                "N": result.N,
                "rho": f"{result.rho:.16g}",
                "rho_uniform": f"{result.rho_uniform:.16g}",
                "improvement_over_uniform": f"{result.improvement_over_uniform:.16g}",
                "search_seconds": f"{result.elapsed_seconds:.9f}",
                "objective_evaluations": result.objective_evaluations,
                "start_count": result.start_count,
                "candidate_count": result.candidate_count,
                "best_source": result.best_source,
                "output_normalization": "min(gap)=1",
                "search_gauge_min_gap": f"{result.search_gauge_min_gap:.16g}",
                "output_scale_from_search_gauge": (
                    f"{result.output_scale_from_search_gauge:.16g}"
                ),
                "lambda_span": f"{result.lambda_span:.16g}",
                "lambda_final": f"{result.lambda_final:.16g}",
                "rescaling_spectral_radius_error": (
                    f"{result.rescaling_spectral_radius_error:.6e}"
                ),
                "min_gap": f"{result.min_gap:.16g}",
                "max_gap": f"{result.max_gap:.16g}",
                "gap_ratio": f"{result.gap_ratio:.16g}",
                "min_gap_count": result.min_gap_count,
                "min_gap_first_1_based": result.min_gap_first_1_based,
                "min_gap_last_1_based": result.min_gap_last_1_based,
                "gap_run_count": result.gap_run_count,
                "gap_pattern": result.gap_pattern,
                "reversal_error": f"{result.reversal_error:.6e}",
                "inward_nonincreasing": result.inward_nonincreasing,
                "max_coordinate_perturbation_gain": (
                    f"{result.max_coordinate_perturbation_gain:.6e}"
                ),
                "max_random_perturbation_gain": (
                    f"{result.max_random_perturbation_gain:.6e}"
                ),
            }
            for j, value in enumerate(result.gaps, start=1):
                row[f"gap_{j}"] = f"{value:.16g}"
            for j, value in enumerate(result.lambdas, start=1):
                row[f"lambda_{j}"] = f"{value:.16g}"
            writer.writerow(row)


def write_json(path: Path, results: Sequence[SearchResult], config: SearchConfig) -> None:
    """Write full-precision min-normalized arrays and the run configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Unrestricted full-dimensional numerical search over all positive "
            "gap coordinates; no symmetry, monotonicity, or jump constraints."
        ),
        "internal_search_gauge": "sum(gaps)=N-1",
        "output_normalization": "min(gaps)=1",
        "lambda_convention": (
            "lambda_1=1 and subsequent lambdas are cumulative output gaps; "
            "lambda_N is generally not N"
        ),
        "global_optimality_claim": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "config": asdict(config),
        "results": [],
    }
    for result in results:
        item = asdict(result)
        item["gaps"] = result.gaps.tolist()
        item["lambdas"] = result.lambdas.tolist()
        payload["results"].append(item)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def run_self_tests() -> None:
    """Check derivatives, scale invariance, and reversal invariance."""
    rng = np.random.default_rng(9157)

    for N in (3, 6, 10):
        problem = FullGapProblem(N)
        u = rng.normal(0.0, 0.7, size=N - 2)
        beta = 64.0
        rho, analytic = problem.value_and_gradient_logits(u, beta=beta)
        numerical = np.empty_like(analytic)
        step = 2e-7
        for j in range(u.size):
            plus = u.copy()
            minus = u.copy()
            plus[j] += step
            minus[j] -= step
            numerical[j] = (
                problem.value_and_gradient_logits(plus, beta=beta)[0]
                - problem.value_and_gradient_logits(minus, beta=beta)[0]
            ) / (2.0 * step)
        error = float(np.max(np.abs(analytic - numerical)))
        if error > 2e-6:
            raise AssertionError(f"gradient test failed for N={N}: {error:g}")

        gaps = problem.logits_to_gaps(u)
        exact = problem.exact_value(gaps)
        scaled = problem.value_and_gradient_gaps(7.3 * gaps, beta=None)[0]
        reversed_value = problem.exact_value(gaps[::-1])
        if abs(exact - scaled) > 2e-11:
            raise AssertionError(f"scale invariance failed for N={N}")
        if abs(exact - reversed_value) > 2e-11:
            raise AssertionError(f"reversal invariance failed for N={N}")

        min_one = problem.normalize_gaps_to_minimum_one(gaps)
        if abs(float(np.min(min_one)) - 1.0) > 8.0 * np.finfo(float).eps:
            raise AssertionError(f"minimum-gap normalization failed for N={N}")
        min_one_value = problem.exact_value(min_one)
        if abs(exact - min_one_value) > 2e-11:
            raise AssertionError(f"minimum-gap rescaling changed rho for N={N}")

    print(
        "Self-tests passed: gradient, scale/reversal invariance, and "
        "min(gap)=1 output normalization."
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unrestricted full-gap numerical search for the B(lambda) problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=48)
    parser.add_argument("--random-starts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--smooth-beta", type=float, default=256.0)
    parser.add_argument("--logit-bound", type=float, default=10.0)
    parser.add_argument("--smooth-maxiter", type=int, default=140)
    parser.add_argument("--exact-maxiter", type=int, default=350)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument(
        "--merge-tolerances",
        type=parse_float_tuple,
        default=(0.008, 0.02, 0.05),
        help="relative adjacent-gap thresholds used for data-driven run refinement",
    )
    parser.add_argument("--validation-restarts", type=int, default=4)
    parser.add_argument("--validation-directions", type=int, default=48)
    parser.add_argument(
        "--thorough",
        action="store_true",
        help="double validation restarts/directions and use a sharper restart soft-min",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "candidates_N3_N48.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "candidates_N3_N48.json",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    if args.n_min < 3 or args.n_max < args.n_min:
        parser.error("require 3 <= n-min <= n-max")
    if args.random_starts < 0:
        parser.error("random-starts must be nonnegative")
    if args.smooth_beta <= 0.0:
        parser.error("smooth-beta must be positive")
    if not 2.0 <= args.logit_bound <= 12.0:
        parser.error("logit-bound must lie in [2,12] for numerical stability")
    if args.shortlist < 1:
        parser.error("shortlist must be positive")

    config = SearchConfig(
        random_starts=args.random_starts,
        seed=args.seed,
        smooth_beta=args.smooth_beta,
        logit_bound=args.logit_bound,
        smooth_maxiter=args.smooth_maxiter,
        exact_maxiter=args.exact_maxiter,
        shortlist=args.shortlist,
        merge_tolerances=args.merge_tolerances,
        validation_restarts=args.validation_restarts,
        random_validation_directions=args.validation_directions,
        thorough=args.thorough,
    )

    results: List[SearchResult] = []
    previous_gaps: np.ndarray | None = None
    print(
        " N       rho(B)        gain vs uniform   sec    starts   evals  "
        "symmetry_err  local_gain  lambda_N   minimum-gap positions   gap pattern"
    )
    for N in range(args.n_min, args.n_max + 1):
        result = search_one_N(N, config, previous_gaps=previous_gaps)
        results.append(result)
        previous_gaps = result.gaps
        print(
            f"{N:2d}  {result.rho:13.10f}  {result.improvement_over_uniform:15.8g}  "
            f"{result.elapsed_seconds:6.2f}  {result.start_count:6d}  "
            f"{result.objective_evaluations:6d}  {result.reversal_error:12.3e}  "
            f"{max(result.max_coordinate_perturbation_gain, result.max_random_perturbation_gain):10.2e}  "
            f"{result.lambda_final:8.3f}  "
            f"{result.min_gap_first_1_based:3d}-{result.min_gap_last_1_based:<3d}  "
            f"{result.gap_pattern}"
        )

        # Write an incremental checkpoint so a long run remains usable if it is
        # interrupted.  The same file is overwritten with all completed rows.
        write_csv(args.csv, results)
        if args.json is not None:
            write_json(args.json, results, config)

    print(f"\nWrote {args.csv}")
    if args.json is not None:
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
