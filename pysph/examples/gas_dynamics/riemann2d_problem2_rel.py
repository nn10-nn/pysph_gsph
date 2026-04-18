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

import os
import sys
import importlib.util
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
        self.hdx = 1.0
        self.nx = 300
        self.ny = 300
        self.dt = 1.0e-4
        self.tf = 0.4
        self.pfreq = 200
        self.nplot = 700

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
        group.add_argument("--nx", action="store", type=int, dest="nx", default=300,
                           help="Number of cells in x (paper uses 300).")
        group.add_argument("--hdx", action="store", type=float, dest="hdx", default=1.0,
                           help="Smoothing-length factor.")
        group.add_argument("--rel-dt", action="store", type=float, dest="rel_dt", default=1.0e-4,
                           help="Fixed time step for this rel case script.")
        group.add_argument("--nplot", action="store", type=int, dest="nplot", default=700,
                           help="Uniform plotting grid resolution for contour rendering.")

    def consume_user_options(self):
        self.nx = int(self.options.nx)
        self.ny = self.nx
        self.hdx = float(self.options.hdx)
        self.dt = float(self.options.rel_dt)
        # `--tf` and `--pfreq` are framework-level options in Application.
        # Use them if provided, otherwise keep defaults from __init__/initialize.
        tf_cli = getattr(self.options, "tf", None)
        if tf_cli is not None:
            self.tf = float(tf_cli)
        pfreq_cli = getattr(self.options, "pfreq", None)
        if pfreq_cli is not None:
            self.pfreq = int(pfreq_cli)
        self.nplot = int(self.options.nplot)

        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.cell_area = self.dx * self.dy

        if self.rank == 0:
            n_particles = self.nx * self.ny
            print(
                "[DEBUG:setup] 2D RP2-rel: nx=%d ny=%d particles=%d hdx=%.3f dt=%.3e tf=%.3f pfreq=%d"
                % (self.nx, self.ny, n_particles, self.hdx, self.dt, self.tf, self.pfreq)
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

        # Paper-style contour: 30 equally spaced levels from -1.38417 to 0.
        levels = np.linspace(-1.38417, 0.0, 30)

        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.1,
        })
        fig, ax = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
        cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.75)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'2D Riemann Problem 2: contour of $\log(\rho)$ at $t=0.4$')

        # Add light major/minor ticks for publication readability.
        ax.minorticks_on()
        ax.tick_params(which='major', direction='in', length=7, width=1.0, top=True, right=True)
        ax.tick_params(which='minor', direction='in', length=4, width=0.8, top=True, right=True)

        png = os.path.join(self.output_dir, "problem4_2_logrho_contour.png")
        pdf = os.path.join(self.output_dir, "problem4_2_logrho_contour.pdf")
        fig.savefig(png, dpi=700, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)

        # Save processed fields for reproducibility.
        np.savez(
            os.path.join(self.output_dir, "problem4_2_logrho_contour_data.npz"),
            x=x, y=y, rho=rho, logrho=logrho, X=X, Y=Y, Z=Z, levels=levels,
            tf=self.tf, gamma=gamma, nx=self.nx, ny=self.ny
        )
        if self.rank == 0:
            print("[DEBUG:plot] saved:", png)
            print("[DEBUG:plot] saved:", pdf)


if __name__ == '__main__':
    app = Riemann2DProblem2Rel()
    app.run()
    app.post_process()
