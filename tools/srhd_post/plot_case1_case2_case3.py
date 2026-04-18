"""Plot three 1D SRHD cases in publication style:
- black solid line: reference (exact)
- dense filled markers: numerical particle solution (uniformly sampled)

Important:
- Sampling is ONLY for visualization clarity.
- Sampling does NOT alter original numerical results and is NOT used for error computation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from tools.srhd_post.case_definitions import CASES
except ModuleNotFoundError:
    # Allow direct execution:
    #   python tools/srhd_post/plot_case1_case2_case3.py
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
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


def _compute_uniform_sample_indices(
    n: int,
    target_points: int | None,
    plot_stride: int | None,
    min_plot_points: int,
    max_plot_points: int,
    score: np.ndarray | None = None,
    focus_fraction: float = 0.45,
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if plot_stride is not None and int(plot_stride) > 1:
        idx = np.arange(0, n, int(plot_stride), dtype=int)
        if idx[-1] != n - 1:
            idx = np.concatenate([idx, np.array([n - 1], dtype=int)])
        return idx

    if target_points is None:
        # Auto-target dense GSPH-looking scatter (not sparse PINN-like markers).
        target_points = int(np.clip(round(n / 8.0), min_plot_points, max_plot_points))
    target_points = int(np.clip(target_points, 1, n))
    if target_points == n:
        return np.arange(n, dtype=int)

    # Deterministic hybrid sampling:
    # - uniform baseline over whole domain
    # - extra points concentrated around high-gradient regions (discontinuities)
    n_focus = int(np.clip(round(target_points * focus_fraction), 0, target_points))
    n_uniform = max(target_points - n_focus, 2)
    idx_uniform = np.unique(np.linspace(0, n - 1, n_uniform).round().astype(int))

    if score is None or n_focus <= 0:
        idx = idx_uniform
    else:
        s = np.asarray(score, dtype=float)
        if s.shape[0] != n:
            raise ValueError(f"score length {s.shape[0]} != n {n}")
        # Candidate region: top 25% score points.
        q = np.quantile(s, 0.75)
        cand = np.where(s >= q)[0]
        if cand.size == 0:
            idx_focus = np.array([], dtype=int)
        elif cand.size <= n_focus:
            idx_focus = cand
        else:
            idx_focus = np.unique(np.linspace(0, cand.size - 1, n_focus).round().astype(int))
            idx_focus = cand[idx_focus]
        idx = np.unique(np.concatenate([idx_uniform, idx_focus]))

    if idx[-1] != n - 1:
        idx = np.concatenate([idx, np.array([n - 1], dtype=int)])
    return np.unique(idx.astype(int))


def _sample_score(num: Dict[str, np.ndarray]) -> np.ndarray:
    x = num["x"]
    eps = 1e-14
    dx = np.maximum(np.gradient(x), eps)
    drho = np.abs(np.gradient(num["rho"]) / dx)
    du = np.abs(np.gradient(num["u"]) / dx)
    dp = np.abs(np.gradient(num["p"]) / dx)

    def _norm(v: np.ndarray) -> np.ndarray:
        vmax = float(np.max(v))
        if vmax <= 0.0:
            return np.zeros_like(v)
        return v / vmax

    # Equal-weight combined indicator.
    return _norm(drho) + _norm(du) + _norm(dp)


def _choose_legend_location(
    ax,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_num: np.ndarray,
    y_num: np.ndarray,
) -> str:
    # Candidate locations ordered from usually clean to fallback.
    candidates = [
        "upper right", "upper left", "lower right", "lower left",
        "center right", "center left", "upper center", "lower center",
    ]

    # Use a moderately sparse subset for robust and fast overlap scoring.
    n_ref = len(x_ref)
    n_num = len(x_num)
    ref_step = max(n_ref // 300, 1)
    num_step = max(n_num // 250, 1)
    ref_pts = np.column_stack((x_ref[::ref_step], y_ref[::ref_step]))
    num_pts = np.column_stack((x_num[::num_step], y_num[::num_step]))
    ref_disp = ax.transData.transform(ref_pts)
    num_disp = ax.transData.transform(num_pts)

    best_loc = candidates[0]
    best_score = None

    for loc in candidates:
        leg = ax.legend(loc=loc, frameon=False)
        ax.figure.canvas.draw()
        bbox = leg.get_window_extent(ax.figure.canvas.get_renderer())

        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        ref_in = (
            (ref_disp[:, 0] >= x0) & (ref_disp[:, 0] <= x1) &
            (ref_disp[:, 1] >= y0) & (ref_disp[:, 1] <= y1)
        )
        num_in = (
            (num_disp[:, 0] >= x0) & (num_disp[:, 0] <= x1) &
            (num_disp[:, 1] >= y0) & (num_disp[:, 1] <= y1)
        )

        # Numerical scatter occlusion is weighted slightly higher.
        score = int(np.count_nonzero(ref_in)) + 2 * int(np.count_nonzero(num_in))
        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc

        leg.remove()

    return best_loc


def _format_gamma(g: float) -> str:
    if abs(g - 5.0 / 3.0) < 1e-12:
        return "5/3"
    if abs(g - 4.0 / 3.0) < 1e-12:
        return "4/3"
    return f"{g:.4g}"


def _plot_one_case(
    case_key: str,
    case_meta: Dict[str, float],
    num: Dict[str, np.ndarray],
    ref: Dict[str, np.ndarray],
    out_dir: Path,
    sample_points: int | None,
    plot_stride: int | None,
    min_plot_points: int,
    max_plot_points: int,
    case3_point_scale: float,
    case3_focus_fraction: float,
    focus_fraction: float,
    num_marker: str,
    num_marker_size: float,
    num_color: str,
    fig_width: float,
    fig_height: float,
    suptitle_fontsize: float,
    suptitle_y: float,
    subplot_top: float,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    score = _sample_score(num)

    # Case 3 often appears over-dense on plateaus; reduce total shown markers
    # while retaining higher density near discontinuities.
    local_min_points = int(min_plot_points)
    local_max_points = int(max_plot_points)
    local_focus_fraction = float(focus_fraction)
    if case_key == "case3":
        local_min_points = max(60, int(round(local_min_points * case3_point_scale)))
        local_max_points = max(local_min_points, int(round(local_max_points * case3_point_scale)))
        local_focus_fraction = float(case3_focus_fraction)

    idx_sample = _compute_uniform_sample_indices(
        n=len(num["x"]),
        target_points=sample_points,
        plot_stride=plot_stride,
        min_plot_points=local_min_points,
        max_plot_points=local_max_points,
        score=score,
        focus_fraction=local_focus_fraction,
    )

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
        "legend.fontsize": 10.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.14, top=subplot_top, wspace=0.15)
    items = [
        ("rho", r"Density ($\rho$)"),
        ("u", r"Velocity ($u$)"),
        ("p", r"Pressure ($p$)"),
    ]
    for ax, (key, title) in zip(axes, items):
        # Reference: black solid line (continuous)
        ax.plot(ref["x"], ref[key], "k-", lw=1.8, label="Reference")
        # Numerical: dense particle-like scatter (uniformly sampled, no line).
        if num_marker == ".":
            ax.plot(
                num["x"][idx_sample], num[key][idx_sample],
                marker=".",
                linestyle="None",
                color=num_color,
                markersize=num_marker_size,
                label="SRHD-GSPH"
            )
        else:
            ax.plot(
                num["x"][idx_sample], num[key][idx_sample],
                marker=num_marker,
                linestyle="None",
                markerfacecolor=num_color,
                markeredgecolor=num_color,
                markeredgewidth=0.0,
                markersize=num_marker_size,
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
        legend_loc = _choose_legend_location(
            ax=ax,
            x_ref=ref["x"], y_ref=ref[key],
            x_num=num["x"][idx_sample], y_num=num[key][idx_sample],
        )
        ax.legend(
            loc=legend_loc,
            frameon=False,
            handlelength=1.9,
            borderaxespad=0.35,
            labelspacing=0.35,
        )

    gamma_str = _format_gamma(case_meta["gamma"])
    fig.suptitle(
        f"{case_meta['label']}: {case_meta['desc']}, "
        f"t = {case_meta['t']}, $\\Gamma$ = {gamma_str}",
        y=suptitle_y,
        fontsize=suptitle_fontsize,
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
                    "and uniformly sampled dense numerical scatter."
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
        help="Force exact sampled-point count for numerical markers. "
             "If omitted, auto dense sampling is used.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=None,
        help="Uniform stride for sampling numerical particles (e.g. 3 means take every 3rd point). "
             "This has priority over --sample-points.",
    )
    parser.add_argument(
        "--min-plot-points",
        type=int,
        default=120,
        help="Lower bound for auto sampled-point count.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=260,
        help="Upper bound for auto sampled-point count.",
    )
    parser.add_argument(
        "--case3-point-scale",
        type=float,
        default=0.62,
        help="Scale factor applied to [min,max]-plot-points for case3 to avoid over-dense plateau markers.",
    )
    parser.add_argument(
        "--case3-focus-fraction",
        type=float,
        default=0.72,
        help="Focus fraction for case3 sampling (higher keeps more points near discontinuities, fewer on plateaus).",
    )
    parser.add_argument(
        "--focus-fraction",
        type=float,
        default=0.45,
        help="Fraction of sampled points concentrated near high-gradient regions.",
    )
    parser.add_argument(
        "--num-marker",
        type=str,
        default=".",
        help="Marker for numerical scatter (default '.').",
    )
    parser.add_argument(
        "--num-marker-size",
        type=float,
        default=4.0,
        help="Marker size for numerical scatter.",
    )
    parser.add_argument(
        "--num-color",
        type=str,
        default="blue",
        help="Color for numerical scatter.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=14.8,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.1,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--suptitle-fontsize",
        type=float,
        default=20.0,
        help="Figure super-title font size.",
    )
    parser.add_argument(
        "--suptitle-y",
        type=float,
        default=0.98,
        help="Figure super-title y location.",
    )
    parser.add_argument(
        "--subplot-top",
        type=float,
        default=0.86,
        help="Top bound for subplot area (leave room for suptitle).",
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
            plot_stride=args.plot_stride,
            min_plot_points=int(args.min_plot_points),
            max_plot_points=int(args.max_plot_points),
            case3_point_scale=float(args.case3_point_scale),
            case3_focus_fraction=float(args.case3_focus_fraction),
            focus_fraction=float(args.focus_fraction),
            num_marker=str(args.num_marker),
            num_marker_size=float(args.num_marker_size),
            num_color=str(args.num_color),
            fig_width=float(args.fig_width),
            fig_height=float(args.fig_height),
            suptitle_fontsize=float(args.suptitle_fontsize),
            suptitle_y=float(args.suptitle_y),
            subplot_top=float(args.subplot_top),
            dpi=args.dpi,
        )

    print("[done] plotting pipeline finished.")


if __name__ == "__main__":
    main()
