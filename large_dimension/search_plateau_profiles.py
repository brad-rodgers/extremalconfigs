#!/usr/bin/env python3
r"""Large-N numerical search for extremal weighted Hilbert matrices.

Problem
=======
For positive gaps ``g_j = lambda_{j+1} - lambda_j`` define

    delta_1 = g_1,
    delta_N = g_{N-1},
    delta_k = min(g_{k-1}, g_k),  2 <= k <= N-1,

and the real skew-symmetric matrix

    B[m,n] = sqrt(delta_m * delta_n) / (lambda_m - lambda_n),  m != n,
    B[m,m] = 0.

The objective is the spectral radius of B, equivalently the largest eigenvalue
of the Hermitian matrix H = 1j*B.  Common rescaling of all gaps leaves H
unchanged, so this program reports gaps with minimum value one.

Large-N structural assumptions
==============================
The search implemented here is *not* over all N-1 gaps.  It assumes:

1. the gap list is reversal-symmetric;
2. it is a contiguous plateau profile, increasing from the centre outwards:

       gamma_L^[n_L], ..., gamma_1^[n_1], 1^[n_0],
       gamma_1^[n_1], ..., gamma_L^[n_L].

Here ``counts = [n_0, ..., n_L]`` and ``levels = [1, gamma_1, ..., gamma_L]``.
The integer counts are searched by adjacent boundary transfers; their proposed
pi-geometric asymptotics are *not* hard-coded as constraints.  Geometric count
profiles may be supplied as one family of initial guesses, together with
continuations from smaller N and other user-supplied profiles.

The program does not require n_{j+1} <= n_j unless ``--require-count-monotone``
is requested.  Strictly increasing gap levels are enforced through positive
log-ratios.

Numerical method
================
A dense matrix costs O(N^2) storage.  Instead, the Cauchy kernel 1/(x-y) is
applied by a one-dimensional hierarchical block method:

* near blocks are evaluated directly;
* a well-separated target/source block uses a source-centred Taylor expansion;
* the reverse block is defined as the negative transpose of the same expansion,
  preserving skew-symmetry exactly in floating point.

The resulting matrix-vector product is approximately O(p N log N), where p is
the expansion rank.  A warm-start Lanczos iteration finds the top eigenpair.

Gap levels are polished using central finite differences of the Rayleigh
quotient at the current eigenvector.  This is deliberately used instead of a
formally analytic squared-kernel sensitivity at very large dynamic range: the
latter can lose many digits through cancellation even while the eigenvalue is
accurate.  At an exact eigenvector,

    d lambda_max(H(t))/dt_j = v^* (dH/dt_j) v,

so finite differences of ``v^* H(t) v`` are first-order correct and require no
extra eigensolves for the derivative itself.

Count refinement is discrete and expensive.  For each adjacent plateau
boundary, the program tries integer transfers in both directions, recomputes
the top eigenpair, and then repolishes the continuous levels for promising
candidates.  Several starts and outer split/merge branches should be used for
serious work.

Numerical validation and limitations
====================================
``validate`` evaluates a supplied vector with several stricter hierarchical
settings and bounds the Taylor truncation error in its Rayleigh quotient block
by block.  ``--direct`` additionally performs an O(N^2)-time, O(N)-auxiliary-
storage direct matrix-vector product, which is practical for occasional checks
up to roughly N=50,000 on a modern workstation.

The search remains a nonconvex local/multistart computation inside the assumed
plateau class.  It does not constitute a proof that a returned profile is the
global extremum, nor a proof of the symmetry/plateau assumptions.

Dependencies: Python 3.10+, NumPy, SciPy, Numba.

Examples
========
Evaluate and validate a stored profile::

    python large_dimension/search_plateau_profiles.py validate \
        --profile large_dimension/witnesses/profile_N8000.json \
        --vector large_dimension/witnesses/eigenvector_N8000.npy \
        --direct --out large_dimension/validation/check_N8000.json

Refine a profile locally::

    python large_dimension/search_plateau_profiles.py search \
        --profile seed_N8000.json --level-iterations 4 --count-cycles 2 \
        --out outputs/refined_N8000.json \
        --vector-out outputs/eigenvector_N8000.npy

Generate a family of noncommittal geometric *starting profiles*::

    python large_dimension/search_plateau_profiles.py seeds --N 8000 --L 8 \
        --q 2.6,2.9,3.141592653589793,3.5 \
        --central-fraction 0.62,0.68,0.74 --out outputs/seeds_N8000.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import numba
import scipy
from numba import njit
from scipy.linalg import eigh_tridiagonal


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Profile representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateauProfile:
    """A symmetric centre-outward plateau gap profile.

    ``counts[j]`` is the number of occurrences of level ``levels[j]`` on one
    side for j>=1.  ``counts[0]`` is the full central plateau count.  Therefore

        counts[0] + 2*sum(counts[1:]) == N-1.
    """

    N: int
    counts: tuple[int, ...]
    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.N < 3:
            raise ValueError("N must be at least 3")
        if len(self.counts) != len(self.levels):
            raise ValueError("counts and levels must have the same length")
        if not self.counts or any(c < 1 for c in self.counts):
            raise ValueError("all plateau counts must be positive integers")
        if self.counts[0] + 2 * sum(self.counts[1:]) != self.N - 1:
            raise ValueError("counts do not sum to N-1 under symmetric expansion")
        if not np.isclose(self.levels[0], 1.0, rtol=0.0, atol=1e-13):
            raise ValueError("levels[0] must be 1 (minimum-gap normalization)")
        if any(not math.isfinite(x) or x <= 0 for x in self.levels):
            raise ValueError("gap levels must be finite and positive")
        if any(self.levels[j + 1] <= self.levels[j] for j in range(len(self.levels) - 1)):
            raise ValueError("gap levels must increase strictly outwards")

    @property
    def L(self) -> int:
        return len(self.counts) - 1

    @property
    def log_ratios(self) -> np.ndarray:
        return np.diff(np.log(np.asarray(self.levels, dtype=float)))

    def to_dict(self) -> dict:
        return {"N": self.N, "counts": list(self.counts), "levels": list(self.levels)}

    @classmethod
    def from_dict(cls, data: dict) -> "PlateauProfile":
        if "profile" in data:
            data = data["profile"]
        counts = data.get("counts", data.get("c"))
        levels = data.get("levels", data.get("lev"))
        if counts is None or levels is None:
            raise KeyError("profile JSON must contain counts/levels (or c/lev)")
        N = int(data.get("N", int(counts[0]) + 2 * sum(map(int, counts[1:])) + 1))
        return cls(N, tuple(map(int, counts)), tuple(map(float, levels)))


def levels_from_log_ratios(t: Sequence[float]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0):
        raise ValueError("all log level ratios must be positive")
    return np.exp(np.r_[0.0, np.cumsum(t)])


def profile_arrays(profile: PlateauProfile) -> tuple[np.ndarray, ...]:
    """Expand a profile into gaps, level labels, positions, deltas and labels.

    Positions are accumulated outwards from the centre.  A conventional
    left-to-right cumulative sum can lose unit central gaps once the outer gap
    scale exceeds about 1e16; centre-outward summation adds nondecreasing terms
    and avoids that cancellation.
    """

    N = profile.N
    counts = np.asarray(profile.counts, dtype=np.int64)
    levels = np.asarray(profile.levels, dtype=float)
    L = profile.L

    gap_labels: list[int] = []
    gaps: list[float] = []
    for level in range(L, 0, -1):
        gap_labels.extend([level] * int(counts[level]))
        gaps.extend([levels[level]] * int(counts[level]))
    gap_labels.extend([0] * int(counts[0]))
    gaps.extend([levels[0]] * int(counts[0]))
    for level in range(1, L + 1):
        gap_labels.extend([level] * int(counts[level]))
        gaps.extend([levels[level]] * int(counts[level]))

    g = np.asarray(gaps, dtype=float)
    qg = np.asarray(gap_labels, dtype=np.int64)
    if g.size != N - 1:
        raise RuntimeError("internal gap expansion error")

    x = np.empty(N, dtype=float)
    middle = N // 2
    x[middle] = 0.0
    for k in range(middle, N - 1):
        x[k + 1] = x[k] + g[k]
    for k in range(middle - 1, -1, -1):
        x[k] = x[k + 1] - g[k]

    q_delta = np.empty(N, dtype=np.int64)
    q_delta[0] = qg[0]
    q_delta[-1] = qg[-1]
    q_delta[1:-1] = np.minimum(qg[:-1], qg[1:])
    delta = levels[q_delta]
    return g, qg, x, delta, q_delta


# ---------------------------------------------------------------------------
# Hierarchical Cauchy matrix-vector product
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("start", "end", "left", "right", "center", "radius", "leaf")

    def __init__(self, start: int, end: int, x: np.ndarray, leaf_size: int, nodes: list) -> None:
        self.start = start
        self.end = end
        self.center = 0.5 * (x[start] + x[end - 1])
        self.radius = 0.5 * (x[end - 1] - x[start])
        self.left = self.right = -1
        self.leaf = end - start <= leaf_size
        nodes.append(self)
        if not self.leaf:
            middle = (start + end) // 2
            self.left = _Node(start, middle, x, leaf_size, nodes)
            self.right = _Node(middle, end, x, leaf_size, nodes)


def _build_partition(
    x: np.ndarray, leaf_size: int = 32, theta: float = 0.22
) -> tuple[np.ndarray, ...]:
    nodes: list[_Node] = []
    _Node(0, len(x), x, leaf_size, nodes)
    index = {id(node): j for j, node in enumerate(nodes)}
    for node in nodes:
        if not node.leaf:
            node.left = index[id(node.left)]
            node.right = index[id(node.right)]

    far: list[tuple[int, int]] = []
    direct: list[tuple[int, int, int]] = []

    def separation_ratio(target: _Node, source: _Node) -> float:
        distance = min(abs(x[target.start] - source.center), abs(x[target.end - 1] - source.center))
        if x[target.start] <= source.center <= x[target.end - 1] or distance == 0.0:
            return math.inf
        return source.radius / distance

    def recurse(a: int, b: int) -> None:
        A, B = nodes[a], nodes[b]
        if a == b:
            if A.leaf:
                direct.append((a, b, 1))
            else:
                recurse(A.left, A.left)
                recurse(A.left, A.right)
                recurse(A.right, A.right)
            return

        ratio_ab = separation_ratio(A, B)
        ratio_ba = separation_ratio(B, A)
        if min(ratio_ab, ratio_ba) <= theta:
            far.append((a, b) if ratio_ab <= ratio_ba else (b, a))
            return
        if A.leaf and B.leaf:
            direct.append((a, b, 0))
            return
        if (not A.leaf) and (B.leaf or A.radius >= B.radius):
            recurse(A.left, b)
            recurse(A.right, b)
        else:
            recurse(a, B.left)
            recurse(a, B.right)

    recurse(0, 0)
    return (
        np.asarray([n.start for n in nodes], dtype=np.int64),
        np.asarray([n.end for n in nodes], dtype=np.int64),
        np.asarray([n.center for n in nodes], dtype=float),
        np.asarray([n.radius for n in nodes], dtype=float),
        np.asarray(far, dtype=np.int64).reshape(-1, 2),
        np.asarray(direct, dtype=np.int64).reshape(-1, 3),
    )


@njit(cache=True)
def _cauchy_mv(
    x: np.ndarray,
    source_values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    far: np.ndarray,
    direct: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Apply the skew Cauchy kernel using direct and Taylor blocks."""

    out = np.zeros(x.size, dtype=np.complex128)
    moments = np.empty(rank, dtype=np.complex128)
    transpose_coefficients = np.empty(rank, dtype=np.complex128)

    for block in range(far.shape[0]):
        target, source = far[block, 0], far[block, 1]
        t0, t1 = starts[target], ends[target]
        s0, s1 = starts[source], ends[source]
        center, radius = centers[source], radii[source]

        for k in range(rank):
            moments[k] = 0.0j
        for j in range(s0, s1):
            u = (x[j] - center) / radius if radius != 0.0 else 0.0
            power = 1.0
            qj = source_values[j]
            for k in range(rank):
                moments[k] += qj * power
                power *= u

        for k in range(rank):
            transpose_coefficients[k] = 0.0j
        for i in range(t0, t1):
            inverse = 1.0 / (x[i] - center)
            ratio = radius * inverse
            power = inverse
            value = 0.0j
            qi = source_values[i]
            for k in range(rank):
                value += moments[k] * power
                transpose_coefficients[k] += qi * power
                power *= ratio
            out[i] += value

        # Use the negative transpose of exactly the same approximation.  This
        # makes the approximate Cauchy matrix skew-symmetric by construction.
        for j in range(s0, s1):
            u = (x[j] - center) / radius if radius != 0.0 else 0.0
            power = 1.0
            value = 0.0j
            for k in range(rank):
                value += transpose_coefficients[k] * power
                power *= u
            out[j] -= value

    for block in range(direct.shape[0]):
        a, b, same = direct[block, 0], direct[block, 1], direct[block, 2]
        a0, a1 = starts[a], ends[a]
        b0, b1 = starts[b], ends[b]
        if same:
            for i in range(a0, a1):
                for j in range(i + 1, a1):
                    kernel = 1.0 / (x[i] - x[j])
                    out[i] += kernel * source_values[j]
                    out[j] -= kernel * source_values[i]
        else:
            for i in range(a0, a1):
                for j in range(b0, b1):
                    kernel = 1.0 / (x[i] - x[j])
                    out[i] += kernel * source_values[j]
                    out[j] -= kernel * source_values[i]
    return out


class HierarchicalHilbertOperator:
    """Matrix-free Hermitian operator H = iB for one profile."""

    def __init__(
        self,
        profile: PlateauProfile,
        leaf_size: int = 32,
        theta: float = 0.22,
        rank: int = 18,
    ) -> None:
        _, _, self.x, self.delta, _ = profile_arrays(profile)
        self.weight = np.sqrt(self.delta)
        self.partition = _build_partition(self.x, leaf_size, theta)
        self.rank = int(rank)
        self.profile = profile

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.complex128)
        if vector.shape != self.x.shape:
            raise ValueError(f"expected vector shape {self.x.shape}, got {vector.shape}")
        return 1j * self.weight * _cauchy_mv(
            self.x, self.weight * vector, *self.partition, self.rank
        )


# ---------------------------------------------------------------------------
# Eigenvalue and direct-validation routines
# ---------------------------------------------------------------------------


def top_eigenpair_lanczos(
    operator: HierarchicalHilbertOperator,
    initial_vector: np.ndarray | None = None,
    tolerance: float = 1e-8,
    max_steps: int = 500,
    check_every: int = 10,
    reorthogonalization_window: int = 7,
) -> tuple[float, np.ndarray, float, int]:
    """Compute the largest eigenpair by warm-start Lanczos iteration."""

    n = operator.profile.N
    if initial_vector is None:
        initial_vector = np.exp(1j * np.pi * (np.arange(n) + 0.5) / n)
    q = np.asarray(initial_vector, dtype=np.complex128).copy()
    if q.shape != (n,) or not np.all(np.isfinite(q)):
        raise ValueError("invalid initial eigenvector")
    q /= np.linalg.norm(q)

    previous = np.zeros(n, dtype=np.complex128)
    beta = 0.0
    basis = np.empty((n, max_steps), dtype=np.complex128)
    diagonal = np.empty(max_steps, dtype=float)
    off_diagonal = np.empty(max_steps - 1, dtype=float)

    for step in range(max_steps):
        basis[:, step] = q
        z = operator.matvec(q)
        if step:
            z -= beta * previous
        diagonal[step] = np.vdot(q, z).real
        z -= diagonal[step] * q
        for k in range(max(0, step - reorthogonalization_window + 1), step):
            z -= np.vdot(basis[:, k], z) * basis[:, k]
        next_beta = float(np.linalg.norm(z))
        if step < max_steps - 1:
            off_diagonal[step] = next_beta

        if (step + 1) % check_every == 0 or step == max_steps - 1:
            values, vectors = eigh_tridiagonal(
                diagonal[: step + 1],
                off_diagonal[:step],
                select="i",
                select_range=(step, step),
            )
            ritz_value = float(values[0])
            coefficients = vectors[:, 0]
            estimated_residual = next_beta * abs(coefficients[-1])
            if estimated_residual <= tolerance or step == max_steps - 1:
                eigenvector = basis[:, : step + 1] @ coefficients
                eigenvector /= np.linalg.norm(eigenvector)
                image = operator.matvec(eigenvector)
                eigenvalue = float(np.vdot(eigenvector, image).real)
                residual = float(np.linalg.norm(image - eigenvalue * eigenvector))
                return eigenvalue, eigenvector, residual, step + 2

        if next_beta < 1e-15:
            break
        previous, q = q, z / next_beta
        beta = next_beta
    raise RuntimeError("Lanczos iteration terminated without an eigenpair")


def direct_matvec(profile: PlateauProfile, vector: np.ndarray, block_rows: int = 128) -> np.ndarray:
    """Apply H exactly in O(N^2) time but only O(block_rows*N) memory."""

    _, _, x, delta, _ = profile_arrays(profile)
    vector = np.asarray(vector, dtype=np.complex128)
    if vector.shape != (profile.N,):
        raise ValueError("vector has the wrong shape")
    weight = np.sqrt(delta)
    weighted_vector = weight * vector
    result = np.empty(profile.N, dtype=np.complex128)
    for first in range(0, profile.N, block_rows):
        last = min(profile.N, first + block_rows)
        denominator = x[first:last, None] - x[None, :]
        rows = np.arange(first, last)
        denominator[np.arange(last - first), rows] = np.inf
        result[first:last] = 1j * weight[first:last] * (weighted_vector[None, :] / denominator).sum(axis=1)
    return result


@njit(cache=True)
def _taylor_rayleigh_bound(
    x: np.ndarray,
    absolute_weighted_vector: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    far: np.ndarray,
    rank: int,
) -> float:
    prefix = np.empty(absolute_weighted_vector.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    for i in range(absolute_weighted_vector.size):
        prefix[i + 1] = prefix[i] + absolute_weighted_vector[i]
    total = 0.0
    for block in range(far.shape[0]):
        target, source = far[block, 0], far[block, 1]
        radius = radii[source]
        if radius == 0.0:
            continue
        source_mass = prefix[ends[source]] - prefix[starts[source]]
        center = centers[source]
        target_bound = 0.0
        for i in range(starts[target], ends[target]):
            distance = abs(x[i] - center)
            ratio = radius / distance
            target_bound += absolute_weighted_vector[i] * ratio**rank / (distance - radius)
        total += 2.0 * source_mass * target_bound
    return total


def validate_vector(
    profile: PlateauProfile,
    vector: np.ndarray,
    configurations: Sequence[tuple[int, float, int]] | None = None,
    do_direct: bool = False,
    direct_block_rows: int = 128,
) -> dict:
    """Return cross-rank Rayleigh checks and Taylor truncation bounds."""

    if configurations is None:
        configurations = ((32, 0.22, 18), (32, 0.18, 20), (16, 0.14, 24), (16, 0.12, 28))
    vector = np.asarray(vector, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    results = []
    for leaf_size, theta, rank in configurations:
        start = time.perf_counter()
        operator = HierarchicalHilbertOperator(profile, leaf_size, theta, rank)
        image = operator.matvec(vector)
        rayleigh = float(np.vdot(vector, image).real)
        residual = float(np.linalg.norm(image - rayleigh * vector))
        absolute_weighted_vector = np.sqrt(operator.delta) * np.abs(vector)
        bound = float(
            _taylor_rayleigh_bound(
                operator.x,
                absolute_weighted_vector,
                *operator.partition[:4],
                operator.partition[4],
                rank,
            )
        )
        results.append(
            {
                "leaf_size": leaf_size,
                "theta": theta,
                "rank": rank,
                "rayleigh": rayleigh,
                "eigen_residual": residual,
                "taylor_rayleigh_error_bound": bound,
                "exact_kernel_rayleigh_lower_bound_ignoring_roundoff": rayleigh - bound,
                "above_pi": rayleigh - bound - math.pi,
                "seconds": time.perf_counter() - start,
            }
        )

    output = {"profile": profile.to_dict(), "hierarchical_checks": results}
    if do_direct:
        start = time.perf_counter()
        image = direct_matvec(profile, vector, direct_block_rows)
        rayleigh = float(np.vdot(vector, image).real)
        output["direct_O_N2_check"] = {
            "rayleigh": rayleigh,
            "eigen_residual": float(np.linalg.norm(image - rayleigh * vector)),
            "above_pi": rayleigh - math.pi,
            "block_rows": direct_block_rows,
            "seconds": time.perf_counter() - start,
        }
    return output


# ---------------------------------------------------------------------------
# Continuous level polishing
# ---------------------------------------------------------------------------


@dataclass
class OperatorSettings:
    leaf_size: int = 32
    theta: float = 0.22
    rank: int = 18
    eigen_tolerance: float = 1e-7
    eigen_steps: int = 500


def solve_profile(
    profile: PlateauProfile,
    settings: OperatorSettings,
    initial_vector: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float, int]:
    operator = HierarchicalHilbertOperator(profile, settings.leaf_size, settings.theta, settings.rank)
    return top_eigenpair_lanczos(
        operator,
        initial_vector,
        tolerance=settings.eigen_tolerance,
        max_steps=settings.eigen_steps,
    )


def rayleigh_at_log_ratios(
    N: int,
    counts: Sequence[int],
    log_ratios: np.ndarray,
    vector: np.ndarray,
    leaf_size: int = 16,
    theta: float = 0.14,
    rank: int = 24,
) -> float:
    levels = levels_from_log_ratios(log_ratios)
    profile = PlateauProfile(N, tuple(map(int, counts)), tuple(map(float, levels)))
    operator = HierarchicalHilbertOperator(profile, leaf_size, theta, rank)
    return float(np.vdot(vector, operator.matvec(vector)).real)


def finite_difference_rayleigh_derivatives(
    profile: PlateauProfile,
    vector: np.ndarray,
    step: float = 1e-2,
    leaf_size: int = 16,
    theta: float = 0.14,
    rank: int = 24,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return Rayleigh value, gradient and diagonal Hessian in log-ratios."""

    t = profile.log_ratios
    base = rayleigh_at_log_ratios(profile.N, profile.counts, t, vector, leaf_size, theta, rank)
    gradient = np.empty_like(t)
    diagonal_hessian = np.empty_like(t)
    for j in range(t.size):
        displacement = np.zeros_like(t)
        displacement[j] = step
        plus = rayleigh_at_log_ratios(
            profile.N, profile.counts, t + displacement, vector, leaf_size, theta, rank
        )
        minus = rayleigh_at_log_ratios(
            profile.N, profile.counts, t - displacement, vector, leaf_size, theta, rank
        )
        gradient[j] = (plus - minus) / (2.0 * step)
        diagonal_hessian[j] = (plus - 2.0 * base + minus) / step**2
    return base, gradient, diagonal_hessian


def polish_levels(
    profile: PlateauProfile,
    settings: OperatorSettings,
    initial_vector: np.ndarray | None = None,
    iterations: int = 3,
    finite_difference_step: float = 1e-2,
    maximum_log_step: float = 2e-2,
    damping: float = 0.7,
) -> tuple[PlateauProfile, float, np.ndarray, float, list[dict]]:
    """Damped diagonal-Newton polishing of all gap-level log ratios."""

    rho, vector, residual, _ = solve_profile(profile, settings, initial_vector)
    history: list[dict] = []
    current = profile
    trust = maximum_log_step

    for iteration in range(iterations):
        start = time.perf_counter()
        rayleigh, gradient, curvature = finite_difference_rayleigh_derivatives(
            current, vector, finite_difference_step
        )
        step = np.zeros_like(gradient)
        reliable = curvature < -1e-10
        step[reliable] = -gradient[reliable] / curvature[reliable]
        # When the local curvature is too flat to estimate reliably, take only
        # a small signed gradient step.
        step[~reliable] = np.sign(gradient[~reliable]) * np.minimum(
            trust, np.abs(gradient[~reliable]) * 100.0
        )
        step = np.clip(damping * step, -trust, trust)

        accepted = False
        best_trial = None
        for line_factor in (1.0, 0.5, 0.25, 0.125):
            trial_t = current.log_ratios + line_factor * step
            if np.any(trial_t <= 1e-8):
                continue
            trial = PlateauProfile(
                current.N,
                current.counts,
                tuple(map(float, levels_from_log_ratios(trial_t))),
            )
            trial_rho, trial_vector, trial_residual, matvecs = solve_profile(trial, settings, vector)
            if best_trial is None or trial_rho > best_trial[0]:
                best_trial = (trial_rho, trial, trial_vector, trial_residual, matvecs, line_factor)
            if trial_rho >= rho - 2e-12:
                accepted = True
                break

        record = {
            "iteration": iteration,
            "rho_before": rho,
            "rayleigh_before": rayleigh,
            "gradient": gradient.tolist(),
            "diagonal_hessian": curvature.tolist(),
            "proposed_step": step.tolist(),
            "accepted": accepted,
            "seconds": time.perf_counter() - start,
        }
        if best_trial is not None:
            record.update(
                {
                    "rho_after": best_trial[0],
                    "line_factor": best_trial[5],
                    "eigen_residual_after": best_trial[3],
                    "matvecs_after": best_trial[4],
                }
            )
        history.append(record)

        if not accepted or best_trial is None:
            trust *= 0.5
            if trust < 2e-4:
                break
            continue
        rho, current, vector, residual = best_trial[:4]
        if np.max(np.abs(best_trial[5] * step)) < 2e-5:
            break

    return current, rho, vector, residual, history


# ---------------------------------------------------------------------------
# Integer count search
# ---------------------------------------------------------------------------


def transfer_counts(counts: Sequence[int], boundary: int, amount: int) -> tuple[int, ...] | None:
    """Transfer gaps across one adjacent plateau boundary.

    At boundary 0, one unit of ``amount`` moves one gap on each side between
    the central plateau and level 1, so n_0 changes by two.  For boundary j>0,
    the transfer is between one-side counts n_j and n_{j+1}.
    """

    result = list(map(int, counts))
    if boundary == 0:
        result[0] -= 2 * amount
        result[1] += amount
    else:
        result[boundary] -= amount
        result[boundary + 1] += amount
    if min(result) < 1:
        return None
    return tuple(result)


def count_is_nonincreasing(counts: Sequence[int]) -> bool:
    return all(counts[j + 1] <= counts[j] for j in range(len(counts) - 1))


def refine_counts(
    profile: PlateauProfile,
    settings: OperatorSettings,
    vector: np.ndarray,
    rho: float,
    cycles: int = 1,
    scan_fraction: float = 0.04,
    level_iterations_after_move: int = 1,
    require_count_monotone: bool = False,
) -> tuple[PlateauProfile, float, np.ndarray, float, list[dict]]:
    """Alternate exact integer boundary trials and continuous level polishing."""

    current, current_rho, current_vector = profile, rho, vector
    _, _, current_residual, _ = solve_profile(current, settings, current_vector)
    history: list[dict] = []

    for cycle in range(cycles):
        changed = False
        for boundary in range(current.L):
            if boundary == 0:
                available = min((current.counts[0] - 1) // 2, current.counts[1] - 1)
                scale = current.counts[1]
            else:
                available = current.counts[boundary] - 1
                scale = max(current.counts[boundary], current.counts[boundary + 1])
            if available < 1:
                continue
            coarse = max(1, min(available, int(round(scan_fraction * scale))))
            trial_amounts = sorted(set((-coarse, -1, 1, coarse)))
            candidates = []
            for amount in trial_amounts:
                counts = transfer_counts(current.counts, boundary, amount)
                if counts is None:
                    continue
                if require_count_monotone and not count_is_nonincreasing(counts):
                    continue
                trial = PlateauProfile(current.N, counts, current.levels)
                trial_rho, trial_vector, trial_residual, matvecs = solve_profile(
                    trial, settings, current_vector
                )
                candidates.append(
                    {
                        "amount": amount,
                        "counts": list(counts),
                        "rho_fixed_levels": trial_rho,
                        "residual": trial_residual,
                        "matvecs": matvecs,
                        "profile": trial,
                        "vector": trial_vector,
                    }
                )

            candidates.sort(key=lambda item: item["rho_fixed_levels"], reverse=True)
            accepted = None
            for candidate in candidates[:2]:
                if candidate["rho_fixed_levels"] < current_rho - 5e-6:
                    continue
                polished, polished_rho, polished_vector, polished_residual, polish_history = polish_levels(
                    candidate["profile"],
                    settings,
                    candidate["vector"],
                    iterations=level_iterations_after_move,
                )
                candidate["rho_polished"] = polished_rho
                candidate["polish_history"] = polish_history
                if polished_rho > current_rho + 1e-10:
                    accepted = (polished, polished_rho, polished_vector, polished_residual, candidate)
                    break

            event = {
                "cycle": cycle,
                "boundary": boundary,
                "rho_before": current_rho,
                "trials": [
                    {k: v for k, v in item.items() if k not in {"profile", "vector", "polish_history"}}
                    for item in candidates
                ],
                "accepted": accepted is not None,
            }
            if accepted is not None:
                current, current_rho, current_vector, current_residual, accepted_candidate = accepted
                event["accepted_amount"] = accepted_candidate["amount"]
                event["rho_after"] = current_rho
                event["counts_after"] = list(current.counts)
                changed = True
            history.append(event)
        if not changed:
            break

    return current, current_rho, current_vector, current_residual, history


def outer_branches(profile: PlateauProfile) -> list[PlateauProfile]:
    """Generate simple outer split and merge alternatives."""

    branches: list[PlateauProfile] = []
    counts, levels = list(profile.counts), list(profile.levels)
    if profile.L >= 2:
        merged_counts = counts[:-2] + [counts[-2] + counts[-1]]
        merged_levels = levels[:-1]
        branches.append(PlateauProfile(profile.N, tuple(merged_counts), tuple(merged_levels)))
    if counts[-1] >= 2:
        ratio = max(1.5, levels[-1] / levels[-2])
        split_counts = counts[:-1] + [counts[-1] - 1, 1]
        split_levels = levels + [levels[-1] * ratio]
        branches.append(PlateauProfile(profile.N, tuple(split_counts), tuple(split_levels)))
    elif profile.L >= 1 and counts[-2] >= 2:
        ratio = max(1.5, levels[-1] / levels[-2])
        split_counts = counts.copy()
        split_counts[-2] -= 1
        split_counts.append(1)
        split_levels = levels + [levels[-1] * ratio]
        branches.append(PlateauProfile(profile.N, tuple(split_counts), tuple(split_levels)))
    return branches


# ---------------------------------------------------------------------------
# Seed generation and output helpers
# ---------------------------------------------------------------------------


def geometric_counts(N: int, L: int, ratio: float, central_fraction: float) -> tuple[int, ...]:
    """Construct a geometric count *seed* without imposing it in optimization."""

    if L < 1 or ratio <= 1 or not (0.05 < central_fraction < 0.98):
        raise ValueError("invalid seed parameters")
    central = max(1, int(round(central_fraction * N)))
    if (N - 1 - central) % 2:
        central += 1 if central < N - 2 else -1
    side_total = (N - 1 - central) // 2
    weights = ratio ** (-np.arange(L, dtype=float))
    raw = side_total * weights / weights.sum()
    side = np.maximum(1, np.rint(raw).astype(int))
    difference = side_total - int(side.sum())
    order = np.argsort(-(raw - np.floor(raw))) if difference > 0 else np.argsort(raw - np.floor(raw))
    cursor = 0
    while difference != 0:
        j = int(order[cursor % L])
        if difference > 0:
            side[j] += 1
            difference -= 1
        elif side[j] > 1:
            side[j] -= 1
            difference += 1
        cursor += 1
        if cursor > 10000:
            raise RuntimeError("could not round geometric counts")
    return (central, *map(int, side))


def generic_levels(N: int, L: int, exponent: float = 0.8) -> tuple[float, ...]:
    """Broad multiscale level seed; it is only an initial condition."""

    # The increment from level j-1 to j is allowed to decrease with depth,
    # but is floored at a small positive value so every seed is admissible.
    increments = np.array(
        [max(0.15, exponent * math.log(N) - 1.3 * j) for j in range(1, L + 1)],
        dtype=float,
    )
    log_levels = np.r_[0.0, np.cumsum(increments)]
    return tuple(map(float, np.exp(log_levels)))


def interpolate_vector(vector: np.ndarray, new_size: int) -> np.ndarray:
    old_size = vector.size
    old_grid = (np.arange(old_size) + 0.5) / old_size
    new_grid = (np.arange(new_size) + 0.5) / new_size
    result = np.interp(new_grid, old_grid, vector.real) + 1j * np.interp(new_grid, old_grid, vector.imag)
    return result / np.linalg.norm(result)


def save_result(
    path: Path,
    profile: PlateauProfile,
    rho: float,
    residual: float,
    level_history: list[dict] | None = None,
    count_history: list[dict] | None = None,
    assumptions: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile.to_dict(),
        "rho": rho,
        "eigen_residual": residual,
        "level_history": level_history or [],
        "count_history": count_history or [],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
        },
        "assumptions": assumptions
        or {
            "reversal_symmetry": True,
            "contiguous_inward_monotone_plateau_levels": True,
            "count_monotonicity_enforced": False,
            "pi_geometric_counts_enforced": False,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_profile(path: Path, index: int = 0) -> PlateauProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        try:
            data = data[index]
        except IndexError as exc:
            raise IndexError(
                f"profile index {index} is outside a list of {len(data)} seeds"
            ) from exc
    if not isinstance(data, dict):
        raise TypeError("profile JSON must contain an object or a list of objects")
    return PlateauProfile.from_dict(data)


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def make_settings(args: argparse.Namespace) -> OperatorSettings:
    return OperatorSettings(args.leaf_size, args.theta, args.rank, args.eigen_tolerance, args.eigen_steps)


def command_search(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile, args.profile_index)
    vector = np.load(args.vector) if args.vector else None
    settings = make_settings(args)
    start = time.perf_counter()

    profile, rho, vector, residual, level_history = polish_levels(
        profile,
        settings,
        vector,
        iterations=args.level_iterations,
        finite_difference_step=args.fd_step,
        maximum_log_step=args.maximum_log_step,
    )
    count_history: list[dict] = []
    if args.count_cycles:
        profile, rho, vector, residual, count_history = refine_counts(
            profile,
            settings,
            vector,
            rho,
            cycles=args.count_cycles,
            scan_fraction=args.count_scan_fraction,
            level_iterations_after_move=args.level_iterations_after_move,
            require_count_monotone=args.require_count_monotone,
        )

    if args.try_outer_branches:
        for branch in outer_branches(profile):
            branch_profile, branch_rho, branch_vector, branch_residual, branch_history = polish_levels(
                branch, settings, vector, iterations=max(1, args.level_iterations_after_move)
            )
            if branch_rho > rho:
                profile, rho, vector, residual = branch_profile, branch_rho, branch_vector, branch_residual
                level_history.extend(branch_history)

    assumptions = {
        "reversal_symmetry": True,
        "contiguous_inward_monotone_plateau_levels": True,
        "count_monotonicity_enforced": bool(args.require_count_monotone),
        "pi_geometric_counts_enforced": False,
        "global_optimality_claim": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_result(args.out, profile, rho, residual, level_history, count_history, assumptions)
    if args.vector_out:
        args.vector_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.vector_out, vector)
    print(json.dumps({"N": profile.N, "rho": rho, "counts": profile.counts, "levels": profile.levels,
                      "eigen_residual": residual, "seconds": assumptions["elapsed_seconds"]}, indent=2))


def command_validate(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile, args.profile_index)
    vector = np.load(args.vector) if args.vector else None
    if vector is None:
        rho, vector, residual, _ = solve_profile(profile, make_settings(args))
        print(f"computed vector: rho={rho:.16g}, residual={residual:.3e}")
    validation = validate_vector(
        profile, vector, do_direct=args.direct, direct_block_rows=args.direct_block_rows
    )
    validation["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


def command_seeds(args: argparse.Namespace) -> None:
    ratios = parse_float_list(args.q)
    central_fractions = parse_float_list(args.central_fraction)
    exponents = parse_float_list(args.level_exponent)
    seeds = []
    for ratio in ratios:
        for central_fraction in central_fractions:
            counts = geometric_counts(args.N, args.L, ratio, central_fraction)
            for exponent in exponents:
                levels = generic_levels(args.N, args.L, exponent)
                try:
                    profile = PlateauProfile(args.N, counts, levels)
                except ValueError:
                    continue
                item = profile.to_dict()
                item["seed_metadata"] = {
                    "count_ratio": ratio,
                    "central_fraction": central_fraction,
                    "level_exponent": exponent,
                    "note": "initial guess only; not an imposed asymptotic",
                }
                seeds.append(item)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    print(f"wrote {len(seeds)} seed profiles to {args.out}")


def add_operator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--theta", type=float, default=0.22)
    parser.add_argument("--rank", type=int, default=18)
    parser.add_argument("--eigen-tolerance", type=float, default=1e-7)
    parser.add_argument("--eigen-steps", type=int, default=500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="locally refine one plateau profile")
    search.add_argument("--profile", type=Path, required=True)
    search.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="index to select when --profile contains a JSON list of seeds",
    )
    search.add_argument("--vector", type=Path)
    search.add_argument("--out", type=Path, required=True)
    search.add_argument("--vector-out", type=Path)
    search.add_argument("--level-iterations", type=int, default=3)
    search.add_argument("--fd-step", type=float, default=1e-2)
    search.add_argument("--maximum-log-step", type=float, default=2e-2)
    search.add_argument("--count-cycles", type=int, default=1)
    search.add_argument("--count-scan-fraction", type=float, default=0.04)
    search.add_argument("--level-iterations-after-move", type=int, default=1)
    search.add_argument("--require-count-monotone", action="store_true")
    search.add_argument("--try-outer-branches", action="store_true")
    add_operator_arguments(search)
    search.set_defaults(function=command_search)

    validate = subparsers.add_parser(
        "validate",
        help="cross-check a profile/vector and optionally do O(N^2) validation",
    )
    validate.add_argument("--profile", type=Path, required=True)
    validate.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="index to select when --profile contains a JSON list of seeds",
    )
    validate.add_argument("--vector", type=Path)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--direct", action="store_true")
    validate.add_argument("--direct-block-rows", type=int, default=128)
    add_operator_arguments(validate)
    validate.set_defaults(function=command_validate)

    seeds = subparsers.add_parser("seeds", help="write a broad family of geometric starting guesses")
    seeds.add_argument("--N", type=int, required=True)
    seeds.add_argument("--L", type=int, required=True)
    seeds.add_argument("--q", default="2.6,2.9,3.141592653589793,3.5")
    seeds.add_argument("--central-fraction", default="0.62,0.68,0.74")
    seeds.add_argument("--level-exponent", default="0.55,0.8,1.0")
    seeds.add_argument("--out", type=Path, required=True)
    seeds.set_defaults(function=command_seeds)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
