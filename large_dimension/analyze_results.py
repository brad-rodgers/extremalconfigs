#!/usr/bin/env python3
r"""Analyze and plot the output of ``search_plateau_profiles.py``.

The script reads the compact JSON file produced for the investigation and
creates the CSV fit table and standalone PNG figures used in the report.  It
uses no hard-coded conjectured count profile when fitting gap levels.  Count
plots distinguish records whose integer counts were actually refined from
records that merely used a pi-geometric count seed.

Usage::

    python large_dimension/analyze_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "large_N_gap_search_results.json"
DEFAULT_FIGURE_DIR = HERE / "figures"
DEFAULT_FIT_OUTPUT = HERE / "results" / "limit_fit_recomputed.csv"


def power_model(N: np.ndarray, limit: float, coefficient: float, exponent: float) -> np.ndarray:
    return limit - coefficient * N ** (-exponent)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def expanded_gaps(record: dict) -> np.ndarray:
    counts = record["counts"]
    levels = record["levels"]
    gaps: list[float] = []
    for j in range(len(counts) - 1, 0, -1):
        gaps.extend([levels[j]] * counts[j])
    gaps.extend([1.0] * counts[0])
    for j in range(1, len(counts)):
        gaps.extend([levels[j]] * counts[j])
    return np.asarray(gaps, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--fit-output", type=Path, default=DEFAULT_FIT_OUTPUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = payload["records"]
    N = np.asarray([item["N"] for item in records], dtype=float)
    rho = np.asarray([item["rho"] for item in records], dtype=float)

    # Main free-power extrapolation.  The cutoff is deliberately in the
    # post-transition regime identified by the plateau search.
    mask = N >= 1700
    parameters, _ = curve_fit(
        power_model,
        N[mask],
        rho[mask],
        p0=(3.1432, 2.4, 0.81),
        bounds=([rho[mask].max(), 0.0, 0.05], [3.2, 100.0, 3.0]),
        maxfev=100000,
    )
    limit, coefficient, exponent = map(float, parameters)

    fit_rows = []
    for cutoff in (800, 1000, 1700, 3000, 5000, 8000):
        selected = N >= cutoff
        p, _ = curve_fit(
            power_model,
            N[selected],
            rho[selected],
            p0=(limit, coefficient, exponent),
            bounds=([rho[selected].max(), 0.0, 0.05], [3.2, 100.0, 3.0]),
            maxfev=100000,
        )
        error = power_model(N[selected], *p) - rho[selected]
        fit_rows.append(
            {
                "N_min": cutoff,
                "point_count": int(selected.sum()),
                "limit": float(p[0]),
                "coefficient": float(p[1]),
                "exponent": float(p[2]),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "max_abs_residual": float(np.max(np.abs(error))),
            }
        )
    args.fit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.fit_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)

    # 1. Spectral-radius convergence.
    grid = np.geomspace(N[N >= 800].min(), N.max() * 1.3, 500)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(N, rho, "o-", label="computed candidates")
    ax.plot(grid, power_model(grid, *parameters), "--", label=fr"fit: $C-aN^{{-p}}$, $p={exponent:.3f}$")
    ax.axhline(math.pi, linestyle=":", label=r"$\pi$")
    ax.axhline(limit, linestyle="-.", label=fr"predicted $C={limit:.7f}$")
    ax.set_xscale("log")
    ax.set_xlabel("matrix size N")
    ax.set_ylabel(r"candidate spectral radius $\rho(B)$")
    ax.set_title("Convergence of the optimized spectral radius")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "rho_convergence.png")

    # 2. Crossing above pi.
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(N, rho - math.pi, "o-")
    ax.axhline(0.0, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("matrix size N")
    ax.set_ylabel(r"$\rho(B)-\pi$")
    ax.set_title(r"The candidate sequence crosses $\pi$ near $N=8000$")
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, args.out_dir / "rho_minus_pi.png")

    # 3. Linearized extrapolation.
    transformed = N[mask] ** (-exponent)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(transformed, rho[mask], "o", label="N >= 1700")
    xline = np.linspace(0.0, transformed.max() * 1.05, 300)
    ax.plot(xline, limit - coefficient * xline, "--", label="least-squares line")
    ax.plot([0.0], [limit], "s", label="intercept / predicted limit")
    ax.set_xlabel(fr"$N^{{-{exponent:.4f}}}$")
    ax.set_ylabel(r"$\rho(B)$")
    ax.set_title("Power-law extrapolation of the limiting spectral radius")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "limit_extrapolation.png")

    # 4. Number of layers.
    layer_count = np.asarray([item["L"] for item in records], dtype=float)
    log_pi_N = np.log(N) / math.log(math.pi)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(log_pi_N, layer_count, "o-", label="observed L")
    line = np.linspace(log_pi_N.min(), log_pi_N.max(), 300)
    ax.plot(line, line, "--", label=r"$L=\log_\pi N$")
    count_constant = 0.5 * (1.0 - 1.0 / math.pi)
    ax.plot(
        line,
        line + math.log(count_constant) / math.log(math.pi),
        ":",
        label=r"$L=\log_\pi(cN)$, $c=(1-1/\pi)/2$",
    )
    ax.set_xlabel(r"$\log_\pi N$")
    ax.set_ylabel("number L of noncentral gap levels")
    ax.set_title("Layer count is consistent with base-pi logarithmic growth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "layer_count_vs_logpiN.png")

    # 5. Count fractions, using only records with independently refined counts.
    refined = [item for item in records if item["count_data_independent_of_conjecture_6"] and item["N"] >= 800]
    a0 = 1.0 - 1.0 / math.pi
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for j in range(0, 6):
        x_values, y_values = [], []
        for item in refined:
            if len(item["counts"]) > j:
                x_values.append(item["N"])
                target = a0 if j == 0 else 0.5 * a0 / math.pi**j
                y_values.append((item["counts"][j] / item["N"]) / target)
        if x_values:
            ax.plot(x_values, y_values, "o-", label=fr"layer $j={j}$")
    ax.axhline(1.0, linestyle=":", label="conjectured asymptotic")
    ax.set_xscale("log")
    ax.set_xlabel("matrix size N")
    ax.set_ylabel("observed count fraction / conjectured count fraction")
    ax.set_title("Tests of the proposed plateau-count constants")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2)
    save_figure(fig, args.out_dir / "count_fraction_ratios.png")

    # 6. Successive count ratios for selected independently searched sizes.
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    selected_sizes = {3000, 5000, 8000, 20000}
    for item in refined:
        if item["N"] not in selected_sizes:
            continue
        counts = item["counts"]
        j = np.arange(1, len(counts) - 1)
        ratios = np.asarray([counts[k] / counts[k + 1] for k in j], dtype=float)
        ax.plot(j, ratios, "o-", label=f"N={item['N']}")
    ax.axhline(math.pi, linestyle=":", label=r"$\pi$")
    ax.set_xlabel("layer index j")
    ax.set_ylabel(r"$n_j/n_{j+1}$")
    ax.set_title("Successive plateau-count ratios")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "successive_count_ratios.png")

    # 7. Fitted power exponents for fixed gap layers.
    fitted_layers, fitted_exponents = [], []
    for j in range(1, 7):
        x_values, y_values = [], []
        for item in records:
            if item["N"] >= 5000 and len(item["levels"]) > j:
                x_values.append(item["N"])
                y_values.append(item["levels"][j])
        if len(x_values) >= 3:
            slope, _ = np.polyfit(np.log(x_values), np.log(y_values), 1)
            fitted_layers.append(j)
            fitted_exponents.append(slope)
    fitted_layers_array = np.asarray(fitted_layers, dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(fitted_layers_array, fitted_exponents, "o-", label="high-N fitted exponent")
    ax.plot(fitted_layers_array, fitted_layers_array, "--", label=r"replacement $\gamma_j=N^{j+o(1)}$")
    ax.plot(fitted_layers_array, 0.5 * fitted_layers_array, ":", label=r"comparison $\gamma_j=N^{j/2+o(1)}$")
    ax.set_xlabel("fixed layer index j")
    ax.set_ylabel("slope of log gamma_j against log N")
    ax.set_title("The observed gap exponents are close to j, not j/2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "gap_power_exponents.png")

    # 8. Scaled fixed-layer levels.
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for j in range(1, 5):
        x_values, y_values = [], []
        for item in records:
            if item["N"] >= 3000 and len(item["levels"]) > j:
                x_values.append(item["N"])
                y_values.append(item["levels"][j] / item["N"] ** j)
        ax.plot(x_values, y_values, "o-", label=fr"$\gamma_{j}/N^{j}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix size N")
    ax.set_ylabel("scaled gap level")
    ax.set_title(r"Fixed layers after scaling by $N^j$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "gap_levels_scaled_by_Nj.png")

    # 9. Selected complete gap profiles.
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for item in records:
        if item["N"] not in {48, 800, 5000, 20000, 50000}:
            continue
        gaps = expanded_gaps(item)
        position = (np.arange(gaps.size) + 0.5) / gaps.size
        ax.plot(position, np.log10(gaps), drawstyle="steps-mid", label=f"N={item['N']}")
    ax.set_xlabel("normalized gap index")
    ax.set_ylabel(r"$\log_{10} g_j$")
    ax.set_title("Evolution of the symmetric multiscale gap profile")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "selected_gap_profiles.png")

    # 10. Central plateau fraction, explicitly distinguishing seeded counts.
    independent = [item for item in records if item["count_data_independent_of_conjecture_6"] and item["N"] >= 800]
    seeded = [item for item in records if not item["count_data_independent_of_conjecture_6"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(
        [item["N"] for item in independent],
        [item["counts"][0] / item["N"] for item in independent],
        "o-",
        label="independently count-refined",
    )
    if seeded:
        ax.plot(
            [item["N"] for item in seeded],
            [item["counts"][0] / item["N"] for item in seeded],
            "x",
            markersize=9,
            label="pi-geometric count seed (not an independent test)",
        )
    ax.axhline(1.0 - 1.0 / math.pi, linestyle=":", label=r"$1-1/\pi$")
    ax.set_xscale("log")
    ax.set_xlabel("matrix size N")
    ax.set_ylabel(r"central plateau fraction $n_0/N$")
    ax.set_title("Central plateau fraction and the proposed asymptotic")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "central_plateau_fraction.png")

    # 11. A direct test of the competing exponents for gamma_1.
    high = [item for item in records if item["N"] >= 3000 and len(item["levels"]) > 1]
    high_N = np.asarray([item["N"] for item in high], dtype=float)
    gamma1 = np.asarray([item["levels"][1] for item in high], dtype=float)
    c_linear = float(np.exp(np.mean(np.log(gamma1[high_N >= 8000] / high_N[high_N >= 8000]))))
    c_sqrt = float(np.exp(np.mean(np.log(gamma1[high_N >= 8000] / np.sqrt(high_N[high_N >= 8000])))))
    line_N = np.geomspace(high_N.min(), high_N.max(), 300)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(high_N, gamma1, "o-", label=r"optimized $\gamma_1$")
    ax.plot(line_N, c_linear * line_N, "--", label=fr"best $cN$ scaling, $c={c_linear:.4f}$")
    ax.plot(line_N, c_sqrt * np.sqrt(line_N), ":", label=r"best $c\sqrt{N}$ scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix size N")
    ax.set_ylabel(r"first noncentral gap level $\gamma_1$")
    ax.set_title(r"The first gap level grows linearly, not as $\sqrt{N}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_figure(fig, args.out_dir / "gamma1_scaling_test.png")

    print(
        json.dumps(
            {
                "limit": limit,
                "coefficient": coefficient,
                "exponent": exponent,
                "plots_written": 11,
                "out_dir": str(args.out_dir),
                "fit_output": str(args.fit_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
