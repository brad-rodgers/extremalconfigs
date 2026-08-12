#!/usr/bin/env python3
r"""Independently validate a stored large-N Rayleigh-quotient witness.

This validator intentionally does *not* import the hierarchical search code. It
uses only NumPy, the compact plateau profile in the stored result table, and a
corresponding normalized vector from ``witnesses/``.

For a profile with gaps g_1,...,g_{N-1}, define

    delta_1 = g_1,
    delta_N = g_{N-1},
    delta_k = min(g_{k-1}, g_k),

and H = iB, where

    B_mn = sqrt(delta_m delta_n)/(lambda_m-lambda_n),  m != n.

For a normalized complex vector v, Hermitian symmetry gives the pairwise form

    v* H v = sum_{m<n} -2 B_mn Im(conj(v_m) v_n).

The code evaluates this expression directly over all N(N-1)/2 pairs.  Terms
are formed in float64, but each block is summed into a NumPy ``longdouble``
accumulator.  It also reports the sum of absolute pair contributions, which is
a useful cancellation/conditioning diagnostic.  This is O(N^2) time and O(block_size^2) auxiliary memory; it is intended for
occasional validation, not for optimization. The computation is floating point,
not an interval-arithmetic proof.

Examples
--------

    python large_dimension/validate_rayleigh_witness.py --N 8000 --extended-terms
    python large_dimension/validate_rayleigh_witness.py --N 50000 \
        --block-size 256 --extended-terms --out outputs/check_N50000.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "large_N_gap_search_results.json"
DEFAULT_WITNESS_DIR = HERE / "witnesses"
PI_DECIMAL = "3.14159265358979323846264338327950288419716939937510"


def load_record(path: Path, N: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    for record in records:
        if int(record["N"]) == N:
            return record
    raise KeyError(f"no record for N={N} in {path}")


def expand_profile(
    record: dict, dtype: np.dtype = np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gaps, center-origin positions, and nearest-neighbor deltas."""
    N = int(record["N"])
    counts = tuple(map(int, record["counts"]))
    levels = tuple(map(float, record["levels"]))
    dtype = np.dtype(dtype).type
    if counts[0] + 2 * sum(counts[1:]) != N - 1:
        raise ValueError("profile counts do not expand to N-1 gaps")
    if not math.isclose(levels[0], 1.0, rel_tol=0.0, abs_tol=1e-13):
        raise ValueError("expected minimum-normalized level gamma_0=1")
    if any(levels[j + 1] <= levels[j] for j in range(len(levels) - 1)):
        raise ValueError("levels must be strictly increasing outwards")

    pieces: list[np.ndarray] = []
    for j in range(len(counts) - 1, 0, -1):
        pieces.append(np.full(counts[j], levels[j], dtype=dtype))
    pieces.append(np.ones(counts[0], dtype=dtype))
    for j in range(1, len(counts)):
        pieces.append(np.full(counts[j], levels[j], dtype=dtype))
    gaps = np.concatenate(pieces)

    # Accumulate from the centre outwards.  This avoids adding unit central
    # gaps after coordinates have reached the scale of the largest outer gap.
    x = np.empty(N, dtype=dtype)
    middle = N // 2
    x[middle] = 0.0
    for k in range(middle, N - 1):
        x[k + 1] = x[k] + gaps[k]
    for k in range(middle - 1, -1, -1):
        x[k] = x[k + 1] - gaps[k]

    delta = np.empty(N, dtype=dtype)
    delta[0] = gaps[0]
    delta[-1] = gaps[-1]
    delta[1:-1] = np.minimum(gaps[:-1], gaps[1:])
    return gaps, x, delta


def pairwise_rayleigh(
    x: np.ndarray,
    delta: np.ndarray,
    vector: np.ndarray,
    block_size: int,
) -> tuple[np.longdouble, np.longdouble]:
    """Evaluate v*Hv by summing every unordered pair exactly once."""
    N = x.size
    if vector.shape != (N,):
        raise ValueError(f"vector shape {vector.shape} does not match N={N}")
    vector = np.asarray(vector, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    weight = np.sqrt(delta)

    total = np.longdouble(0.0)
    absolute_total = np.longdouble(0.0)
    for i0 in range(0, N, block_size):
        i1 = min(N, i0 + block_size)
        xi = x[i0:i1]
        wi = weight[i0:i1]
        vi = vector[i0:i1]

        # Strict upper triangle inside the diagonal block.
        cross_phase = np.imag(np.conj(vi)[:, None] * vi[None, :])
        denominator = xi[:, None] - xi[None, :]
        denominator[np.diag_indices(i1 - i0)] = np.inf
        coefficient = wi[:, None] * wi[None, :] / denominator
        contribution = -2.0 * coefficient * cross_phase
        upper = np.triu_indices(i1 - i0, 1)
        values = contribution[upper]
        total += np.sum(values, dtype=np.longdouble)
        absolute_total += np.sum(np.abs(values), dtype=np.longdouble)

        # Complete off-diagonal blocks; every pair has i < j here.
        for j0 in range(i1, N, block_size):
            j1 = min(N, j0 + block_size)
            xj = x[j0:j1]
            wj = weight[j0:j1]
            vj = vector[j0:j1]
            cross_phase = np.imag(np.conj(vi)[:, None] * vj[None, :])
            coefficient = wi[:, None] * wj[None, :] / (xi[:, None] - xj[None, :])
            values = -2.0 * coefficient * cross_phase
            total += np.sum(values, dtype=np.longdouble)
            absolute_total += np.sum(np.abs(values), dtype=np.longdouble)
    return total, absolute_total



def pairwise_rayleigh_extended(
    x: np.ndarray,
    delta: np.ndarray,
    vector: np.ndarray,
    block_size: int,
) -> tuple[np.longdouble, np.longdouble]:
    """As :func:`pairwise_rayleigh`, but form every term in longdouble.

    Real and imaginary parts are handled separately because extended-precision
    complex arithmetic is less consistently optimized across NumPy builds.
    """
    x = np.asarray(x, dtype=np.longdouble)
    delta = np.asarray(delta, dtype=np.longdouble)
    N = x.size
    if vector.shape != (N,):
        raise ValueError(f"vector shape {vector.shape} does not match N={N}")
    real = np.asarray(vector.real, dtype=np.longdouble)
    imag = np.asarray(vector.imag, dtype=np.longdouble)
    norm = np.sqrt(np.sum(real * real + imag * imag, dtype=np.longdouble))
    real /= norm
    imag /= norm
    weight = np.sqrt(delta)

    total = np.longdouble(0.0)
    absolute_total = np.longdouble(0.0)
    for i0 in range(0, N, block_size):
        i1 = min(N, i0 + block_size)
        xi, wi = x[i0:i1], weight[i0:i1]
        ar, ai = real[i0:i1], imag[i0:i1]

        cross_phase = ar[:, None] * ai[None, :] - ai[:, None] * ar[None, :]
        denominator = xi[:, None] - xi[None, :]
        denominator[np.diag_indices(i1 - i0)] = np.inf
        contribution = -2 * (wi[:, None] * wi[None, :] / denominator) * cross_phase
        upper = np.triu_indices(i1 - i0, 1)
        values = contribution[upper]
        total += np.sum(values, dtype=np.longdouble)
        absolute_total += np.sum(np.abs(values), dtype=np.longdouble)

        for j0 in range(i1, N, block_size):
            j1 = min(N, j0 + block_size)
            xj, wj = x[j0:j1], weight[j0:j1]
            br, bi = real[j0:j1], imag[j0:j1]
            cross_phase = ar[:, None] * bi[None, :] - ai[:, None] * br[None, :]
            values = -2 * (wi[:, None] * wj[None, :] / (xi[:, None] - xj[None, :])) * cross_phase
            total += np.sum(values, dtype=np.longdouble)
            absolute_total += np.sum(np.abs(values), dtype=np.longdouble)
    return total, absolute_total

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
    )
    parser.add_argument(
        "--vector",
        type=Path,
        default=None,
        help="witness vector; defaults to witnesses/eigenvector_N<N>.npy",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--extended-terms", action="store_true",
        help="form pair contributions in NumPy longdouble as well as accumulating in longdouble",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.N < 3 or args.block_size < 2:
        parser.error("require N >= 3 and block-size >= 2")

    record = load_record(args.results, args.N)
    profile_dtype = np.longdouble if args.extended_terms else np.float64
    _, x, delta = expand_profile(record, profile_dtype)
    vector_path = args.vector or (DEFAULT_WITNESS_DIR / f"eigenvector_N{args.N}.npy")
    if not vector_path.exists():
        raise FileNotFoundError(
            f"witness vector not found: {vector_path}. Supply --vector explicitly."
        )
    vector = np.asarray(np.load(vector_path), dtype=np.complex128)

    started = time.perf_counter()
    evaluator = pairwise_rayleigh_extended if args.extended_terms else pairwise_rayleigh
    rayleigh, absolute_sum = evaluator(x, delta, vector, args.block_size)
    elapsed = time.perf_counter() - started
    pi_ld = np.longdouble(PI_DECIMAL)
    output = {
        "N": args.N,
        "method": (
            "direct all-pairs Hermitian pair sum with longdouble terms"
            if args.extended_terms
            else "direct all-pairs Hermitian pair sum with float64 terms"
        ),
        "pair_count": args.N * (args.N - 1) // 2,
        "block_size": args.block_size,
        "results_file": str(args.results),
        "vector_file": str(vector_path),
        "rayleigh_longdouble_accumulator": str(rayleigh),
        "rayleigh_float": float(rayleigh),
        "above_pi": float(rayleigh - pi_ld),
        "sum_absolute_pair_contributions": float(absolute_sum),
        "absolute_sum_over_rayleigh": float(absolute_sum / abs(rayleigh)),
        "vector_norm": float(np.linalg.norm(vector)),
        "stored_candidate_rho": float(record["rho"]),
        "difference_from_stored_candidate": float(rayleigh - np.longdouble(str(record["rho"]))),
        "seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "longdouble_epsilon": str(np.finfo(np.longdouble).eps),
            "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
            "float64_epsilon": str(np.finfo(np.float64).eps),
        },
        "roundoff_note": (
            ("Pair terms and accumulation use NumPy longdouble. " if args.extended_terms else
             "Pair terms are float64; block and global accumulation use longdouble. ")
            + "This is a numerical cross-check, not an interval-arithmetic proof."
        ),
    }
    text = json.dumps(output, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
