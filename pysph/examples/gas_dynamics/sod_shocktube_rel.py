"""1D SRHD Sod shock tube (paper setup) with standalone GSPHRelScheme."""

import os
import sys
import importlib.util
from pathlib import Path
import numpy
from math import sqrt

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

from pysph.base.nnps import DomainManager
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
try:
    from pysph.sph.gas_dynamics.scheme_rel import GSPHRelScheme
except ModuleNotFoundError:
    # Fallback: load module directly from file path to avoid package-resolution
    # issues under debugger/alternate PYTHONPATH setups.
    scheme_file = _repo_root / "pysph" / "sph" / "gas_dynamics" / "scheme_rel.py"
    spec = importlib.util.spec_from_file_location("scheme_rel_local", str(scheme_file))
    if spec is None or spec.loader is None:
        raise
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    GSPHRelScheme = mod.GSPHRelScheme


dim = 1
gamma = 5.0/3.0
gamma1 = gamma - 1.0
dt = 1e-5
tf = 0.5


class SodShockTubeRel(ShockTubeSetup):
    def __init__(self, *args, **kwargs):
        super(SodShockTubeRel, self).__init__(*args, **kwargs)
        # Safe defaults so create_scheme() can be called before
        # consume_user_options().
        self.hdx = 1.0
        self.nl = 800
        self.dscheme = "constant_volume"
        # Debug controls for step-by-step diagnostics.
        self._dbg_init_done = False
        self._dbg_first_step_pre_done = False
        self._dbg_first_step_post_done = False
        self._dbg_probe_orig_id = None
        self._dbg_pre_probe = {}

    def initialize(self):
        # Computational domain is extended to reduce boundary-kernel artifacts.
        # Physical window for comparison remains x in [0, 1].
        self.xmin = -0.5
        self.xmax = 1.5
        self.plot_xmin = 0.0
        self.plot_xmax = 1.0
        self.x0 = 0.5
        self.rhol = 1.0
        self.rhor = 0.125
        self.pl = 1.0
        self.pr = 0.1
        self.ul = 0.5
        self.ur = 0.5

    def add_user_options(self, group):
        group.add_argument("--hdx", action="store", type=float, dest="hdx", default=1.0)
        group.add_argument("--nl", action="store", type=float, dest="nl", default=800)
        group.add_argument(
            "--dscheme", choices=["constant_mass", "constant_volume"],
            dest="dscheme", default="constant_volume"
        )

    def consume_user_options(self):
        self.nl = int(self.options.nl)
        self.hdx = self.options.hdx
        self.dscheme = self.options.dscheme
        if self.rank == 0:
            print(
                "[DEBUG:resolution] using nl=%d, hdx=%.3f (target: reduce right-shock ringing and sharpen discontinuities)"
                % (self.nl, self.hdx)
            )
        self.dxl = (self.x0 - self.xmin) / self.nl
        if self.dscheme == 'constant_mass':
            ratio = self.rhor / self.rhol
            self.dxr = self.dxl / ratio
        else:
            self.dxr = self.dxl
        self.h0 = self.hdx * self.dxr
        self.dt = dt
        self.tf = tf

    def create_particles(self):
        f, b = self.generate_particles(
            xmin=self.xmin, xmax=self.xmax, x0=self.x0, rhol=self.rhol,
            rhor=self.rhor, pl=self.pl, pr=self.pr, bx=0.00, gamma1=gamma1,
            ul=self.ul, ur=self.ur, dxl=self.dxl, dxr=self.dxr, h0=self.h0
        )
        self.scheme.setup_properties([f, b])

        # Build SRHD conservative variables from primitive variables.
        rho0 = f.rho.copy()     # rest-mass density
        p = f.p
        u = f.u
        e = f.e
        Gamma = 1.0 / numpy.sqrt(1.0 - numpy.clip(u*u, 0.0, 1.0 - 1e-12))
        D = rho0 * Gamma
        dx = f.m / numpy.maximum(rho0, 1e-14)
        f.m[:] = D * dx
        f.rho[:] = D
        f.rho_rest[:] = rho0
        f.gamma_rel[:] = Gamma
        hhat = 1.0 + e + p / numpy.maximum(rho0, 1e-14)
        f.hhat[:] = hhat
        f.qx[:] = hhat * Gamma * u
        f.qy[:] = 0.0
        f.qz[:] = 0.0
        f.ehat[:] = hhat * Gamma - p / numpy.maximum(D, 1e-14)
        f.chi[:] = p / numpy.maximum(D, 1e-14)
        f.gamma_ad[:] = gamma

        # Debug: validate initialized states near x=0.25, 0.5, 0.75.
        if self.rank == 0 and (not self._dbg_init_done):
            print("\n[DEBUG:init] Problem-1 target states:")
            print("  Left : rho=1.0, u=0.5, p=1.0")
            print("  Right: rho=0.125, u=0.5, p=0.1")
            targets = [0.25, 0.50, 0.75]
            for xt in targets:
                idx = int(numpy.argmin(numpy.abs(f.x - xt)))
                side = "left" if f.x[idx] <= self.x0 else "right"
                expected = "left-state" if xt < self.x0 else (
                    "right-state" if xt > self.x0 else "discontinuity-neighborhood"
                )
                print(
                    "[DEBUG:init] probe x*=%.3f -> x=%.6f (%s, expect %s): "
                    "rho_rest=%.6e, u=%.6e, p=%.6e, D=%.6e, q=%.6e, ehat=%.6e"
                    % (
                        xt, f.x[idx], side, expected,
                        f.rho_rest[idx], f.u[idx], f.p[idx], f.rho[idx],
                        f.qx[idx], f.ehat[idx]
                    )
                )
            self._dbg_init_done = True

        return [f]

    def create_domain(self):
        # Open boundaries; physical region of interest is interior [0, 1].
        return None

    def configure_scheme(self):
        self.scheme.configure_solver(tf=self.tf, dt=self.dt)

    def create_scheme(self):
        kf = getattr(self, "hdx", 1.2)
        # Explicitly lock the principal Riemann solver to Rusanov.
        rsolver = 0
        if self.rank == 0:
            print("[DEBUG:solver] Using SRHD Riemann solver = Rusanov (rsolver=%d)" % rsolver)
        return GSPHRelScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kf, rsolver=rsolver, niter=20, tol=1e-8
        )

    def pre_step(self, solver):
        # Debug only for first physical step.
        if self.rank > 0 or self._dbg_first_step_pre_done:
            return
        pa = self.particles[0]
        # Probe near discontinuity for stage diagnostics.
        idx_mid = int(numpy.argmin(numpy.abs(pa.x - self.x0)))
        self._dbg_probe_orig_id = int(pa.orig_idx[idx_mid])
        self._dbg_pre_probe = {
            "x": float(pa.x[idx_mid]),
            "D": float(pa.rho[idx_mid]),
            "h": float(pa.h[idx_mid]),
            "rho_rest": float(pa.rho_rest[idx_mid]),
            "u": float(pa.u[idx_mid]),
            "p": float(pa.p[idx_mid]),
            "D_min": float(pa.rho.min()),
            "D_max": float(pa.rho.max()),
            "h_min": float(pa.h.min()),
            "h_max": float(pa.h.max()),
        }
        print(
            "\n[DEBUG:step1:pre] t=%.6e dt=%.6e probe(orig_idx=%d): "
            "x=%.6e, D=%.6e, h=%.6e, rho_rest=%.6e, u=%.6e, p=%.6e"
            % (
                solver.t, solver.dt, self._dbg_probe_orig_id,
                self._dbg_pre_probe["x"], self._dbg_pre_probe["D"],
                self._dbg_pre_probe["h"], self._dbg_pre_probe["rho_rest"],
                self._dbg_pre_probe["u"], self._dbg_pre_probe["p"]
            )
        )
        self._dbg_first_step_pre_done = True

    def post_step(self, solver):
        # Debug only for first physical step.
        if self.rank > 0 or self._dbg_first_step_post_done:
            return
        pa = self.particles[0]

        idxs = numpy.where(pa.orig_idx == self._dbg_probe_orig_id)[0]
        if len(idxs) > 0:
            idx = int(idxs[0])
        else:
            idx = int(numpy.argmin(numpy.abs(pa.x - self.x0)))

        print(
            "[DEBUG:step1:post] t=%.6e dt=%.6e probe(orig_idx=%d): "
            "q(after cons)=%.6e, ehat(after cons)=%.6e, "
            "u(after rec)=%.6e, p(after rec)=%.6e, x(after pos)=%.6e"
            % (
                solver.t, solver.dt, int(pa.orig_idx[idx]),
                pa.qx[idx], pa.ehat[idx], pa.u[idx], pa.p[idx], pa.x[idx]
            )
        )

        # Check if D/h were recomputed within this same step.
        D_pre = self._dbg_pre_probe["D"]
        h_pre = self._dbg_pre_probe["h"]
        D_post = float(pa.rho[idx])
        h_post = float(pa.h[idx])
        D_same = abs(D_post - D_pre) <= 1e-14 * max(1.0, abs(D_pre))
        h_same = abs(h_post - h_pre) <= 1e-14 * max(1.0, abs(h_pre))
        print(
            "[DEBUG:step1:Dh] within-step recompute check: "
            "D_pre=%.6e, D_post=%.6e, h_pre=%.6e, h_post=%.6e"
            % (D_pre, D_post, h_pre, h_post)
        )
        if D_same and h_same:
            print("[DEBUG:step1:Dh] D/h unchanged in this step; D/h are recomputed in equation groups of the next solver cycle.")
        else:
            print("[DEBUG:step1:Dh] D/h changed within this step.")

        # Step-1 end global diagnostics.
        rho_min = float(pa.rho_rest.min())
        rho_max = float(pa.rho_rest.max())
        u_min = float(pa.u.min())
        u_max = float(pa.u.max())
        p_min = float(pa.p.min())
        p_max = float(pa.p.max())
        D_min = float(pa.rho.min())
        D_max = float(pa.rho.max())
        q_min = float(pa.qx.min())
        q_max = float(pa.qx.max())
        ehat_min = float(pa.ehat.min())
        ehat_max = float(pa.ehat.max())
        print(
            "[DEBUG:step1:range] rho_rest[min,max]=[%.6e, %.6e], "
            "u[min,max]=[%.6e, %.6e], p[min,max]=[%.6e, %.6e]"
            % (rho_min, rho_max, u_min, u_max, p_min, p_max)
        )
        print(
            "[DEBUG:step1:range] D[min,max]=[%.6e, %.6e], "
            "q(min,max)=[%.6e, %.6e], ehat[min,max]=[%.6e, %.6e]"
            % (D_min, D_max, q_min, q_max, ehat_min, ehat_max)
        )

        bad_rho = numpy.any(pa.rho_rest <= 0.0)
        bad_p = numpy.any(pa.p <= 0.0)
        bad_u = numpy.any(numpy.abs(pa.u) >= 1.0)
        print(
            "[DEBUG:step1:phys] any(rho<=0)=%s, any(p<=0)=%s, any(|u|>=1)=%s"
            % (str(bool(bad_rho)), str(bool(bad_p)), str(bool(bad_u)))
        )

        self._dbg_first_step_post_done = True

    def post_process(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0 or len(self.output_files) == 0:
            return

        from pysph.solver.utils import load
        data = load(self.output_files[-1])
        pa = data['arrays']['fluid']

        # 1) Sort by x and clip to physical window.
        idx = numpy.argsort(pa.x)
        x_sorted = pa.x[idx]
        D_sorted = pa.rho[idx]
        rho_sorted = pa.rho_rest[idx]
        u_sorted = pa.u[idx]
        p_sorted = pa.p[idx]
        mask = (x_sorted >= self.plot_xmin) & (x_sorted <= self.plot_xmax)
        x = x_sorted[mask]
        D = D_sorted[mask]
        rho0 = rho_sorted[mask]
        u = u_sorted[mask]
        p = p_sorted[mask]

        # Ensure unique/monotone abscissa before interpolation.
        x_uni, uni_idx = numpy.unique(x, return_index=True)
        rho_uni = rho0[uni_idx]
        u_uni = u[uni_idx]
        p_uni = p[uni_idx]
        D_uni = D[uni_idx]

        # 2) Interpolate to a uniform fine grid for publication-quality curves.
        n_plot = 1000
        x_plot = numpy.linspace(self.plot_xmin, self.plot_xmax, n_plot)
        rho_plot = numpy.interp(x_plot, x_uni, rho_uni)
        u_plot = numpy.interp(x_plot, x_uni, u_uni)
        p_plot = numpy.interp(x_plot, x_uni, p_uni)
        D_plot = numpy.interp(x_plot, x_uni, D_uni)

        # 3) Optional reference overlay interface (npz with keys: x, rho, u, p).
        ref = None
        ref_candidates = [
            os.environ.get("SRHD_REF_NPZ", ""),
            os.path.join(self.output_dir, "reference_rel.npz"),
            os.path.join(self.output_dir, "exact_rel.npz"),
        ]
        for fref in ref_candidates:
            if fref and os.path.isfile(fref):
                try:
                    ref_npz = numpy.load(fref)
                    if all(k in ref_npz for k in ["x", "rho", "u", "p"]):
                        r_idx = numpy.argsort(ref_npz["x"])
                        xr = ref_npz["x"][r_idx]
                        rr = ref_npz["rho"][r_idx]
                        ur = ref_npz["u"][r_idx]
                        pr = ref_npz["p"][r_idx]
                        mref = (xr >= self.plot_xmin) & (xr <= self.plot_xmax)
                        ref = dict(x=xr[mref], rho=rr[mref], u=ur[mref], p=pr[mref], file=fref)
                        print("[DEBUG:plot] reference overlay enabled from:", fref)
                        break
                except Exception as exc:
                    print("[DEBUG:plot] failed to read reference npz:", fref, "error:", exc)

        # 4) Unified, publication-style 1x3 figure.
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.1,
        })
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        series = [
            (axes[0], rho_plot, r"Density $\rho$", ref["rho"] if ref else None),
            (axes[1], u_plot, r"Velocity $u$", ref["u"] if ref else None),
            (axes[2], p_plot, r"Pressure $p$", ref["p"] if ref else None),
        ]
        for ax, y, ylab, yref in series:
            ax.plot(x_plot, y, color="#1f77b4", lw=2.2, label="SRHD-GSPH")
            if ref is not None:
                ax.plot(ref["x"], yref, color="black", lw=1.8, ls="--", label="Reference")
            ymin = float(y.min())
            ymax = float(y.max())
            if ref is not None and len(yref) > 0:
                ymin = min(ymin, float(numpy.min(yref)))
                ymax = max(ymax, float(numpy.max(yref)))
            pad = 0.06 * max(ymax - ymin, 1e-12)
            ax.set_xlim(self.plot_xmin, self.plot_xmax)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("x")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.22, lw=0.7)
            ax.legend(loc="best", frameon=True, framealpha=0.9)

        png_file = os.path.join(self.output_dir, "profiles_1d_publication.png")
        pdf_file = os.path.join(self.output_dir, "profiles_1d_publication.pdf")
        fig.savefig(png_file, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_file, bbox_inches="tight")
        plt.close(fig)
        print("[DEBUG:plot] saved:", png_file)
        print("[DEBUG:plot] saved:", pdf_file)

        fname = os.path.join(self.output_dir, 'results_rel.npz')
        numpy.savez(
            fname,
            # metadata
            t=self.tf, gamma=gamma, xmin=self.plot_xmin, xmax=self.plot_xmax, x0=self.x0,
            # primitive
            x=x, u=u, p=p, rho_rest=rho0,
            x_plot=x_plot, rho_plot=rho_plot, u_plot=u_plot, p_plot=p_plot,
            e=pa.e[idx][mask], cs=pa.cs[idx][mask],
            hhat=pa.hhat[idx][mask], gamma_rel=pa.gamma_rel[idx][mask],
            # conservative
            D=D, D_plot=D_plot, qx=pa.qx[idx][mask], qy=pa.qy[idx][mask], qz=pa.qz[idx][mask],
            ehat=pa.ehat[idx][mask], chi=pa.chi[idx][mask]
        )


if __name__ == '__main__':
    app = SodShockTubeRel()
    app.run()
    app.post_process()
