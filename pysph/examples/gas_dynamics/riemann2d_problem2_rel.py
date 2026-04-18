"""2D SRHD Riemann Problem 2 (Yang & Tang 2012, Example 4.3 / Fig. 4.2).

Goal:
- Reproduce the density-log contour at t=0.4 for the 2D relativistic
  Riemann problem 2 (interaction of four rarefaction waves).

Reference setup (V = (rho, u, v, p)^T):
- x in [0, 1], y in [0, 1], discontinuities at x=0.5 and y=0.5
- gamma = 5/3, t_end = 0.4
- Q1 (x>0.5, y>0.5): rho=1.0,    u=0.0,     v=0.0,     p=1.0
- Q2 (x<0.5, y>0.5): rho=0.5771, u=-0.3529, v=0.0,     p=0.4
- Q3 (x<0.5, y<0.5): rho=1.0,    u=-0.3529, v=-0.3529, p=1.0
- Q4 (x>0.5, y<0.5): rho=0.5771, u=0.0,     v=-0.3529, p=0.4

Notes:
- This script adds an independent 2D case file and does not modify core
  algorithm modules.
- Rusanov (rsolver=0) is explicitly selected for robustness.
"""

import json
import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
import numpy as np

# Ensure local source tree is preferred before any pysph imports.
_repo_root = Path(__file__).resolve().parents[3]
_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

# If a non-local pysph was already imported, clear it to avoid mixed imports.
if 'pysph' in sys.modules:
    _loaded = getattr(sys.modules['pysph'], '__file__', '') or ''
    if _repo_root_str not in _loaded:
        _to_del = [k for k in list(sys.modules.keys())
                   if k == 'pysph' or k.startswith('pysph.')]
        for _k in _to_del:
            del sys.modules[_k]

# Force serial mode to avoid optional MPI/Zoltan import path on environments
# where parallel extensions are not compiled.
import pysph  # noqa: E402
pysph._in_parallel = False
pysph._has_zoltan = False
pysph._has_mpi = False

from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application

try:
    from pysph.sph.gas_dynamics.scheme_rel import GSPHRelScheme
except ModuleNotFoundError:
    scheme_file = _repo_root / "pysph" / "sph" / "gas_dynamics" / "scheme_rel.py"
    spec = importlib.util.spec_from_file_location("scheme_rel_local", str(scheme_file))
    if spec is None or spec.loader is None:
        raise
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    GSPHRelScheme = mod.GSPHRelScheme


dim = 2
gamma = 5.0 / 3.0
gamma1 = gamma - 1.0


class Riemann2DProblem2Rel(Application):
    def __init__(self, *args, **kwargs):
        super(Riemann2DProblem2Rel, self).__init__(*args, **kwargs)
        # Safe defaults because create_scheme() may be called before
        # consume_user_options().
        self.hdx = 1.12
        self.nx = 320
        self.ny = 320
        self.dt = 7.0e-5
        self.cfl_rel = 0.18
        self.tf = 0.4
        self.pfreq = 100
        self.nplot = 900
        self.nlevels = 30
        self.logrho_level_min = -1.38417
        self.logrho_level_max = 0.0
        self.plot_smooth_passes = 1
        self.export_fig_dir = "postprocess/figures"

    def initialize(self):
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0
        self.x0 = 0.5
        self.y0 = 0.5

        # Physical end time from reference figure.
        self.tf = 0.4

    def add_user_options(self, group):
        group.add_argument("--nx", action="store", type=int, dest="nx", default=320,
                           help="Number of cells in x (paper uses 300).")
        group.add_argument("--hdx", action="store", type=float, dest="hdx", default=1.12,
                           help="Smoothing-length factor.")
        group.add_argument("--rel-dt", action="store", type=float, dest="rel_dt", default=None,
                           help="Optional fixed dt override. If omitted, use conservative CFL estimate.")
        group.add_argument("--cfl-rel", action="store", type=float, dest="cfl_rel", default=0.18,
                           help="CFL coefficient for dt estimate when --rel-dt is not set.")
        group.add_argument("--nplot", action="store", type=int, dest="nplot", default=900,
                           help="Uniform plotting grid resolution for contour rendering.")
        group.add_argument("--nlevels", action="store", type=int, dest="nlevels", default=30,
                           help="Number of contour levels for log(rho).")
        group.add_argument("--logrho-level-min", action="store", type=float,
                           dest="logrho_level_min", default=-1.38417,
                           help="Minimum contour level for log(rho).")
        group.add_argument("--logrho-level-max", action="store", type=float,
                           dest="logrho_level_max", default=0.0,
                           help="Maximum contour level for log(rho).")
        group.add_argument("--plot-smooth-passes", action="store", type=int,
                           dest="plot_smooth_passes", default=1,
                           help="Display-only smoothing passes on gridded log(rho).")
        group.add_argument("--export-fig-dir", action="store", type=str,
                           dest="export_fig_dir", default="postprocess/figures",
                           help="Extra directory (relative to repo root) to copy final publication figures.")

    def consume_user_options(self):
        self.nx = int(self.options.nx)
        self.ny = self.nx
        self.hdx = float(self.options.hdx)
        self.cfl_rel = float(self.options.cfl_rel)
        rel_dt_opt = getattr(self.options, "rel_dt", None)
        # `--tf` and `--pfreq` are framework-level options in Application.
        # Use them if provided, otherwise keep defaults from __init__/initialize.
        tf_cli = getattr(self.options, "tf", None)
        if tf_cli is not None:
            self.tf = float(tf_cli)
        pfreq_cli = getattr(self.options, "pfreq", None)
        if pfreq_cli is not None:
            self.pfreq = int(pfreq_cli)
        self.nplot = int(self.options.nplot)
        self.nlevels = max(5, int(self.options.nlevels))
        self.logrho_level_min = float(self.options.logrho_level_min)
        self.logrho_level_max = float(self.options.logrho_level_max)
        self.plot_smooth_passes = max(0, int(self.options.plot_smooth_passes))
        self.export_fig_dir = str(self.options.export_fig_dir)

        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.cell_area = self.dx * self.dy

        # Conservative initial CFL estimate if fixed dt is not explicitly set.
        if rel_dt_opt is not None and float(rel_dt_opt) > 0.0:
            self.dt = float(rel_dt_opt)
            dt_mode = "fixed"
        else:
            # Estimate max signal speed from the four initial quadrant states.
            # This is a stable startup estimate (not a dynamic CFL update).
            quad_states = [
                (1.0, 0.0, 0.0, 1.0),
                (0.5771, -0.3529, 0.0, 0.4),
                (1.0, -0.3529, -0.3529, 1.0),
                (0.5771, 0.0, -0.3529, 0.4),
            ]
            max_sig = 1.0e-8
            for rho_s, u_s, v_s, p_s in quad_states:
                e_s = p_s / (gamma1 * max(rho_s, 1e-14))
                h_s = 1.0 + e_s + p_s / max(rho_s, 1e-14)
                cs2_s = gamma * p_s / max(rho_s * h_s, 1e-14)
                cs2_s = min(max(cs2_s, 0.0), 1.0 - 1e-12)
                cs_s = cs2_s**0.5
                vel_s = (u_s*u_s + v_s*v_s)**0.5
                max_sig = max(max_sig, min(0.999999, vel_s + cs_s))
            self.dt = self.cfl_rel * min(self.dx, self.dy) / max_sig
            dt_mode = "cfl_estimated"

        if self.rank == 0:
            n_particles = self.nx * self.ny
            print(
                "[DEBUG:setup] 2D RP2-rel: nx=%d ny=%d particles=%d hdx=%.3f dt=%.3e(%s) tf=%.3f pfreq=%d "
                "nplot=%d smooth_passes=%d"
                % (
                    self.nx, self.ny, n_particles, self.hdx, self.dt, dt_mode, self.tf,
                    self.pfreq, self.nplot, self.plot_smooth_passes
                )
            )

    def _set_quadrant_primitive(self, x, y):
        # Vectorized quadrant assignment for performance and consistency.
        rho = np.empty_like(x)
        u = np.empty_like(x)
        v = np.empty_like(x)
        p = np.empty_like(x)

        right = x > self.x0
        top = y > self.y0

        q1 = right & top            # x>0.5, y>0.5
        q2 = (~right) & top         # x<=0.5, y>0.5
        q3 = (~right) & (~top)      # x<=0.5, y<=0.5
        q4 = right & (~top)         # x>0.5, y<=0.5

        rho[q1], u[q1], v[q1], p[q1] = 1.0, 0.0, 0.0, 1.0
        rho[q2], u[q2], v[q2], p[q2] = 0.5771, -0.3529, 0.0, 0.4
        rho[q3], u[q3], v[q3], p[q3] = 1.0, -0.3529, -0.3529, 1.0
        rho[q4], u[q4], v[q4], p[q4] = 0.5771, 0.0, -0.3529, 0.4
        return rho, u, v, p

    def create_particles(self):
        # Cell-centered uniform mesh.
        xg, yg = np.mgrid[
            self.xmin + 0.5*self.dx:self.xmax:self.dx,
            self.ymin + 0.5*self.dy:self.ymax:self.dy
        ]
        x = xg.ravel()
        y = yg.ravel()

        rho0, u, v, p = self._set_quadrant_primitive(x, y)

        # Primitive-derived quantities.
        e = p / (gamma1 * np.maximum(rho0, 1e-14))
        hhat = 1.0 + e + p / np.maximum(rho0, 1e-14)
        v2 = u*u + v*v
        v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
        Gamma = 1.0 / np.sqrt(1.0 - v2)
        D = rho0 * Gamma
        chi = p / np.maximum(D, 1e-14)
        qx = hhat * Gamma * u
        qy = hhat * Gamma * v
        qz = np.zeros_like(qx)
        ehat = hhat * Gamma - chi

        # In this SRHD-GSPH implementation, mass is tied to lab density D.
        m = D * self.cell_area
        h = np.ones_like(x) * (self.hdx * max(self.dx, self.dy))

        pa = gpa(
            name='fluid',
            x=x, y=y, z=np.zeros_like(x),
            u=u, v=v, w=np.zeros_like(x),
            m=m, rho=D, p=p, e=e, h=h, h0=h.copy()
        )
        self.scheme.setup_properties([pa])

        # Populate SRHD fields expected by rel scheme/equations.
        pa.rho[:] = D
        pa.rho_rest[:] = rho0
        pa.gamma_rel[:] = Gamma
        pa.hhat[:] = hhat
        pa.qx[:] = qx
        pa.qy[:] = qy
        pa.qz[:] = qz
        pa.ehat[:] = ehat
        pa.chi[:] = chi
        pa.gamma_ad[:] = gamma

        if self.rank == 0:
            print("[DEBUG:init] quadrant states assigned for Problem 4.3 (2D RP2).")
            probes = [(0.75, 0.75), (0.25, 0.75), (0.25, 0.25), (0.75, 0.25)]
            for (xp, yp) in probes:
                idx = int(np.argmin((pa.x - xp)**2 + (pa.y - yp)**2))
                print(
                    "[DEBUG:init] probe (%.2f,%.2f) -> rho=%.5f u=%.5f v=%.5f p=%.5f D=%.5f"
                    % (xp, yp, pa.rho_rest[idx], pa.u[idx], pa.v[idx], pa.p[idx], pa.rho[idx])
                )

        return [pa]

    def create_domain(self):
        # Open domain for this independent SRHD benchmark script.
        return None

    def create_scheme(self):
        # Explicitly select Rusanov for robustness.
        rsolver = 0
        kf = getattr(self, "hdx", 1.0)
        if self.rank == 0:
            print("[DEBUG:solver] Using SRHD Riemann solver = Rusanov (rsolver=%d)" % rsolver)
        return GSPHRelScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kf, rsolver=rsolver, niter=20, tol=1e-8
        )

    def configure_scheme(self):
        self.scheme.configure_solver(tf=self.tf, dt=self.dt, pfreq=self.pfreq)

    def _interpolate_logrho_on_grid(self, x, y, logrho):
        # High-quality contour rendering with robust fallback.
        import matplotlib.tri as mtri

        nplot = max(200, int(self.nplot))
        gx = np.linspace(self.xmin, self.xmax, nplot)
        gy = np.linspace(self.ymin, self.ymax, nplot)
        X, Y = np.meshgrid(gx, gy)

        try:
            tri = mtri.Triangulation(x, y)
            interp = mtri.LinearTriInterpolator(tri, logrho)
            Z = interp(X, Y)
            Z = np.array(Z, dtype=float)
            # Fill potential NaNs by nearest-bin average as fallback.
            bad = ~np.isfinite(Z)
            if np.any(bad):
                num, xe, ye = np.histogram2d(
                    x, y, bins=nplot, range=[[self.xmin, self.xmax], [self.ymin, self.ymax]],
                    weights=logrho
                )
                den, _, _ = np.histogram2d(
                    x, y, bins=nplot, range=[[self.xmin, self.xmax], [self.ymin, self.ymax]]
                )
                zb = np.full((nplot, nplot), np.nan)
                m = den > 0.0
                zb[m] = num[m] / den[m]
                # histogram2d gives [xbin, ybin], transpose to [y, x].
                zb = zb.T
                Z[bad] = np.nanmean(zb) if np.isfinite(np.nanmean(zb)) else -10.0
            return X, Y, Z
        except Exception as exc:
            if self.rank == 0:
                print("[DEBUG:plot] triangulation interpolation fallback:", exc)
            num, xe, ye = np.histogram2d(
                x, y, bins=nplot, range=[[self.xmin, self.xmax], [self.ymin, self.ymax]],
                weights=logrho
            )
            den, _, _ = np.histogram2d(
                x, y, bins=nplot, range=[[self.xmin, self.xmax], [self.ymin, self.ymax]]
            )
            Z = np.full((nplot, nplot), np.nan)
            m = den > 0.0
            Z[m] = num[m] / den[m]
            Z = Z.T
            X, Y = np.meshgrid(
                0.5*(xe[:-1] + xe[1:]),
                0.5*(ye[:-1] + ye[1:])
            )
            return X, Y, Z

    def _smooth2d_nanaware(self, Z, passes=1):
        """Display-only light smoothing for contour readability."""
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
                Znew = np.where(wgt > 0.0, acc / wgt, np.nan)
            Zs = Znew
        return Zs

    def _save_figure(self, fig, name, dpi=600):
        png = os.path.join(self.output_dir, f"{name}.png")
        pdf = os.path.join(self.output_dir, f"{name}.pdf")
        fig.savefig(png, dpi=dpi, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')

        export_dir = (_repo_root / self.export_fig_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        png2 = export_dir / f"{name}.png"
        pdf2 = export_dir / f"{name}.pdf"
        fig.savefig(str(png2), dpi=dpi, bbox_inches='tight')
        fig.savefig(str(pdf2), bbox_inches='tight')

        if self.rank == 0:
            print("[DEBUG:plot] saved:", png)
            print("[DEBUG:plot] saved:", pdf)
            print("[DEBUG:plot] exported:", str(png2))
            print("[DEBUG:plot] exported:", str(pdf2))

        return {
            "output_png": png,
            "output_pdf": pdf,
            "export_png": str(png2),
            "export_pdf": str(pdf2),
        }

    def post_process(self):
        if self.rank > 0 or len(self.output_files) == 0:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return

        from pysph.solver.utils import load
        data = load(self.output_files[-1])
        pa = data['arrays']['fluid']

        # Prefer rest density for log(rho), consistent with paper notation.
        rho = np.array(pa.rho_rest, copy=False)
        x = np.array(pa.x, copy=False)
        y = np.array(pa.y, copy=False)

        # Clip to physical window.
        mask = (
            (x >= self.xmin) & (x <= self.xmax) &
            (y >= self.ymin) & (y <= self.ymax)
        )
        x = x[mask]
        y = y[mask]
        rho = rho[mask]

        logrho = np.log(np.maximum(rho, 1e-14))
        X, Y, Z = self._interpolate_logrho_on_grid(x, y, logrho)
        Zs = self._smooth2d_nanaware(Z, passes=self.plot_smooth_passes)

        levels = np.linspace(
            float(self.logrho_level_min), float(self.logrho_level_max), int(self.nlevels)
        )

        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.1,
        })
        # Main publication figure: clean contour lines (paper-friendly).
        fig, ax = plt.subplots(figsize=(8.4, 7.8), constrained_layout=True)
        ax.contour(X, Y, Zs, levels=levels, colors='black', linewidths=0.78)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'2D Riemann problem: contour of $\log(\rho)$ at $t=0.4$')

        # Add light major/minor ticks for publication readability.
        ax.minorticks_on()
        ax.tick_params(which='major', direction='in', length=7, width=1.0, top=True, right=True)
        ax.tick_params(which='minor', direction='in', length=4, width=0.8, top=True, right=True)
        main_saved = self._save_figure(fig, "fig5-4-case4-2d", dpi=700)
        plt.close(fig)

        # Diagnostics 1: raw contour (before display smoothing).
        fig1, ax1 = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
        ax1.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.65)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(self.xmin, self.xmax)
        ax1.set_ylim(self.ymin, self.ymax)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(r'Diagnostic: raw contour of $\log(\rho)$')
        diag1_saved = self._save_figure(fig1, "diag_case4_2d_logrho_contour_raw", dpi=500)
        plt.close(fig1)

        # Diagnostics 2: filled contour for structure readability.
        fig2, ax2 = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
        cf = ax2.contourf(X, Y, Zs, levels=levels, cmap='cividis', extend='both')
        ax2.contour(X, Y, Zs, levels=levels, colors='k', linewidths=0.35, alpha=0.35)
        cbar = fig2.colorbar(cf, ax=ax2, fraction=0.048, pad=0.03)
        cbar.set_label(r'$\log(\rho)$')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlim(self.xmin, self.xmax)
        ax2.set_ylim(self.ymin, self.ymax)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(r'Diagnostic: filled contour of $\log(\rho)$')
        diag2_saved = self._save_figure(fig2, "diag_case4_2d_logrho_contourf", dpi=500)
        plt.close(fig2)

        # Diagnostics 3: pcolormesh view to inspect interpolation smoothness.
        fig3, ax3 = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
        pcm = ax3.pcolormesh(X, Y, Zs, shading='auto', cmap='cividis')
        cbar3 = fig3.colorbar(pcm, ax=ax3, fraction=0.048, pad=0.03)
        cbar3.set_label(r'$\log(\rho)$')
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_xlim(self.xmin, self.xmax)
        ax3.set_ylim(self.ymin, self.ymax)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title(r'Diagnostic: pcolormesh of $\log(\rho)$')
        diag3_saved = self._save_figure(fig3, "diag_case4_2d_logrho_pcolormesh", dpi=500)
        plt.close(fig3)

        valid_raw = np.isfinite(Z)
        valid_s = np.isfinite(Zs)

        run_params = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "script": str(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "case_name": "2D Riemann problem",
            "domain": [float(self.xmin), float(self.xmax), float(self.ymin), float(self.ymax)],
            "x0": float(self.x0),
            "y0": float(self.y0),
            "gamma": float(gamma),
            "tf": float(self.tf),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "hdx": float(self.hdx),
            "dt": float(self.dt),
            "cfl_rel": float(self.cfl_rel),
            "pfreq": int(self.pfreq),
            "nplot": int(self.nplot),
            "nlevels": int(self.nlevels),
            "logrho_level_min": float(self.logrho_level_min),
            "logrho_level_max": float(self.logrho_level_max),
            "plot_smooth_passes": int(self.plot_smooth_passes),
            "rho_rest_min": float(np.min(rho)),
            "rho_rest_max": float(np.max(rho)),
            "logrho_min": float(np.min(logrho)),
            "logrho_max": float(np.max(logrho)),
            "grid_raw_valid_ratio": float(np.count_nonzero(valid_raw) / valid_raw.size),
            "grid_smooth_valid_ratio": float(np.count_nonzero(valid_s) / valid_s.size),
            "main_figure": main_saved,
            "diagnostics": [diag1_saved, diag2_saved, diag3_saved],
        }

        params_json = os.path.join(self.output_dir, "case4_2d_run_params.json")
        with open(params_json, "w", encoding="utf-8") as fjson:
            json.dump(run_params, fjson, indent=2, ensure_ascii=False)

        report_txt = os.path.join(self.output_dir, "case4_2d_run_report.txt")
        with open(report_txt, "w", encoding="utf-8") as frep:
            frep.write("2D SRHD-GSPH final run report\n")
            frep.write("=" * 44 + "\n")
            for k in [
                "timestamp", "case_name", "gamma", "tf", "nx", "ny", "dx", "dy",
                "hdx", "dt", "cfl_rel", "pfreq", "nplot", "nlevels",
                "logrho_level_min", "logrho_level_max", "plot_smooth_passes",
                "rho_rest_min", "rho_rest_max", "logrho_min", "logrho_max",
                "grid_raw_valid_ratio", "grid_smooth_valid_ratio"
            ]:
                frep.write(f"{k}: {run_params[k]}\n")
            frep.write("\nmain figure:\n")
            frep.write(json.dumps(main_saved, indent=2, ensure_ascii=False) + "\n")
            frep.write("\ndiagnostics:\n")
            frep.write(json.dumps(run_params["diagnostics"], indent=2, ensure_ascii=False) + "\n")

        # Save processed fields for reproducibility.
        np.savez(
            os.path.join(self.output_dir, "case4_2d_plot_fields.npz"),
            x=x, y=y, rho=rho, logrho=logrho, X=X, Y=Y, Z_raw=Z, Z_smooth=Zs, levels=levels,
            tf=self.tf, gamma=gamma, nx=self.nx, ny=self.ny, dt=self.dt, hdx=self.hdx
        )

        if self.rank == 0:
            print("[DEBUG:report] params json:", params_json)
            print("[DEBUG:report] text report:", report_txt)
            print("[DEBUG:data] saved:", os.path.join(self.output_dir, "case4_2d_plot_fields.npz"))


if __name__ == '__main__':
    app = Riemann2DProblem2Rel()
    app.run()
    app.post_process()
