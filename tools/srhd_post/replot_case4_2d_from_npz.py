"""Re-plot 2D case-4 figure from saved npz fields (no solver rerun required)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _smooth2d_nanaware(Z, passes=1):
    if passes <= 0:
        return np.array(Z, copy=True)
    Zs = np.array(Z, dtype=float, copy=True)
    for _ in range(int(passes)):
        valid = np.isfinite(Zs)
        data = np.where(valid, Zs, 0.0)
        data_pad = np.pad(data, ((1, 1), (1, 1)), mode='edge')
        valid_pad = np.pad(valid.astype(float), ((1, 1), (1, 1)), mode='edge')
        acc = np.zeros_like(data, dtype=float)
        wgt = np.zeros_like(data, dtype=float)
        for i in range(3):
            for j in range(3):
                d = data_pad[i:i + data.shape[0], j:j + data.shape[1]]
                v = valid_pad[i:i + data.shape[0], j:j + data.shape[1]]
                acc += d * v
                wgt += v
        with np.errstate(invalid='ignore', divide='ignore'):
            Zs = np.where(wgt > 0.0, acc / wgt, np.nan)
    return Zs


def _save(fig, out_dir: Path, name: str, dpi: int = 600):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[ok] saved {png}")
    print(f"[ok] saved {pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot case4 2D publication/diagnostic figures from saved npz fields."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to case4_2d_plot_fields.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("postprocess/figures"),
        help="Output directory for replotted figures.",
    )
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--nlevels", type=int, default=30)
    parser.add_argument("--logrho-level-min", type=float, default=-1.38417)
    parser.add_argument("--logrho-level-max", type=float, default=0.0)
    parser.add_argument("--dpi", type=int, default=700)
    args = parser.parse_args()

    z = np.load(args.input)
    X = np.asarray(z["X"], dtype=float)
    Y = np.asarray(z["Y"], dtype=float)
    Z_raw = np.asarray(z["Z_raw"], dtype=float)
    tf = float(z["tf"]) if "tf" in z else 0.4

    Z = _smooth2d_nanaware(Z_raw, passes=max(0, int(args.smooth_passes)))
    levels = np.linspace(float(args.logrho_level_min), float(args.logrho_level_max), int(args.nlevels))

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.1,
    })

    # Main figure.
    fig, ax = plt.subplots(figsize=(8.4, 7.8), constrained_layout=True)
    ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.78)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"2D Riemann problem: contour of $\log(\rho)$ at $t={tf:.3g}$")
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", length=7, width=1.0, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=4, width=0.8, top=True, right=True)
    _save(fig, args.out_dir, "fig5-4-case4-2d", dpi=args.dpi)
    plt.close(fig)

    # Diagnostic contourf.
    fig2, ax2 = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
    cf = ax2.contourf(X, Y, Z, levels=levels, cmap="cividis", extend="both")
    ax2.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.35, alpha=0.35)
    cbar = fig2.colorbar(cf, ax=ax2, fraction=0.048, pad=0.03)
    cbar.set_label(r"$\log(\rho)$")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    ax2.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(r"Diagnostic: filled contour of $\log(\rho)$")
    _save(fig2, args.out_dir, "diag_case4_2d_logrho_contourf", dpi=500)
    plt.close(fig2)

    # Diagnostic pcolormesh.
    fig3, ax3 = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
    pcm = ax3.pcolormesh(X, Y, Z, shading="auto", cmap="cividis")
    cbar3 = fig3.colorbar(pcm, ax=ax3, fraction=0.048, pad=0.03)
    cbar3.set_label(r"$\log(\rho)$")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    ax3.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title(r"Diagnostic: pcolormesh of $\log(\rho)$")
    _save(fig3, args.out_dir, "diag_case4_2d_logrho_pcolormesh", dpi=500)
    plt.close(fig3)

    print("[done] replot completed (no solver rerun).")


if __name__ == "__main__":
    main()

