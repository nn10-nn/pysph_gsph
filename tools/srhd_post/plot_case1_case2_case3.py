"""Plot three 1D SRHD cases in publication style:
- black solid line: reference (exact)
- blue hollow circles: numerical particle solution (uniformly sampled)

Important:
- Sampling is ONLY for visualization clarity.
- Sampling does NOT alter original numerical results and is NOT used for error computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from tools.srhd_post.case_definitions import CASES


def _load_num_result(path: Path, x_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Numerical result file not found: {path}")
    z = np.load(path)
    required = ["x", "rho_rest", "u", "p"]
    for key in required:
        if key not in z:
            raise KeyError(f"Missing '{key}' in numerical npz: {path}")

    x = np.asarray(z["x"], dtype=float)
    rho = np.asarray(z["rho_rest"], dtype=float)
    u = np.asarray(z["u"], dtype=float)
    p = np.asarray(z["p"], dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    rho = rho[idx]
    u = u[idx]
    p = p[idx]

    xmin, xmax = x_range
    mask = (x >= xmin) & (x <= xmax)
    x = x[mask]
    rho = rho[mask]
    u = u[mask]
    p = p[mask]

    return {
        "x": x,
        "rho": rho,
        "u": u,
        "p": p,
        "t": float(z["t"]) if "t" in z else np.nan,
    }


def _load_ref_result(path: Path, x_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Reference result file not found: {path}")
    z = np.load(path)
    required = ["x", "rho", "u", "p"]
    for key in required:
        if key not in z:
            raise KeyError(f"Missing '{key}' in reference npz: {path}")

    x = np.asarray(z["x"], dtype=float)
    rho = np.asarray(z["rho"], dtype=float)
    u = np.asarray(z["u"], dtype=float)
    p = np.asarray(z["p"], dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    rho = rho[idx]
    u = u[idx]
    p = p[idx]

    xmin, xmax = x_range
    mask = (x >= xmin) & (x <= xmax)
    x = x[mask]
    rho = rho[mask]
    u = u[mask]
    p = p[mask]

    return {
        "x": x,
        "rho": rho,
        "u": u,
        "p": p,
        "t": float(z["t"]) if "t" in z else np.nan,
    }


def _compute_uniform_sample_indices(n: int, target_points: int | None) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if target_points is None:
        # Auto-target 40~80 points according to total particle count.
        target_points = int(np.clip(round(n / 20.0), 40, 80))
    target_points = int(np.clip(target_points, 1, n))
    if target_points == n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, target_points).round().astype(int))


def _plot_one_case(
    case_key: str,
    case_meta: Dict[str, float],
    num: Dict[str, np.ndarray],
    ref: Dict[str, np.ndarray],
    out_dir: Path,
    sample_points: int | None,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    idx_sample = _compute_uniform_sample_indices(len(num["x"]), sample_points)

    # Consistency checks (time/range).
    t_num = num.get("t", np.nan)
    t_ref = ref.get("t", np.nan)
    if np.isfinite(t_num) and np.isfinite(t_ref):
        dt_abs = abs(t_num - t_ref)
        if dt_abs > 1e-10:
            print(
                f"[warn] {case_key}: output time mismatch num={t_num:.8g}, ref={t_ref:.8g}, "
                f"|Δt|={dt_abs:.3e}"
            )

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    items = [
        ("rho", r"Density ($\rho$)"),
        ("u", r"Velocity ($u$)"),
        ("p", r"Pressure ($p$)"),
    ]
    for ax, (key, title) in zip(axes, items):
        # Reference: black solid line (continuous)
        ax.plot(ref["x"], ref[key], "k-", lw=1.8, label="Reference")
        # Numerical: blue hollow circles (uniformly sampled)
        ax.plot(
            num["x"][idx_sample], num[key][idx_sample],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="blue",
            markeredgewidth=1.1,
            markersize=4.8,
            label="SRHD-GSPH"
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_xlim(case_meta["x1"], case_meta["x2"])
        y_all = np.concatenate([ref[key], num[key]])
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = 0.08 * max(y_max - y_min, 1e-12)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(False)
        ax.legend(loc="best", frameon=False)

    fig.suptitle(
        f"{case_meta['label']}: {case_meta['desc']}  "
        f"(t={case_meta['t']}, $\\Gamma$={case_meta['gamma']})",
        y=1.02
    )

    out_png = out_dir / f"{case_meta['fig_prefix']}.png"
    out_pdf = out_dir / f"{case_meta['fig_prefix']}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {case_key}: saved {out_png}")
    print(f"[ok] {case_key}: saved {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot three 1D SRHD cases with exact reference (black line) "
                    "and uniformly sampled numerical points (blue hollow circles)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root path.",
    )
    parser.add_argument(
        "--exact-dir",
        type=Path,
        default=Path("postprocess/exact_riemann"),
        help="Directory containing {case}_exact.npz generated by generate_exact_solutions.py",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("postprocess/figures"),
        help="Output directory for final chapter-ready figures.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["case1", "case2", "case3"],
        help="Subset of cases to plot.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=None,
        help="Force number of sampled points for numerical markers. "
             "Default: auto in [40, 80].",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for PNG outputs.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    exact_dir = (repo_root / args.exact_dir).resolve()
    fig_dir = (repo_root / args.fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    for key in args.cases:
        if key not in CASES:
            raise KeyError(f"Unknown case '{key}'. Valid keys: {list(CASES.keys())}")
        meta = CASES[key]
        num_path = (repo_root / meta["num_npz_rel"]).resolve()
        ref_path = (exact_dir / f"{key}_exact.npz").resolve()

        print(f"[case] {key}")
        print(f"  numerical: {num_path}")
        print(f"  reference: {ref_path}")

        x_range = (meta["x1"], meta["x2"])
        num = _load_num_result(num_path, x_range)
        ref = _load_ref_result(ref_path, x_range)
        _plot_one_case(
            case_key=key,
            case_meta=meta,
            num=num,
            ref=ref,
            out_dir=fig_dir,
            sample_points=args.sample_points,
            dpi=args.dpi,
        )

    print("[done] plotting pipeline finished.")


if __name__ == "__main__":
    main()

