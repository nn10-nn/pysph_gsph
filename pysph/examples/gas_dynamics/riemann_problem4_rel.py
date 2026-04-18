"""1D SRHD Riemann Problem 4 (Reflection test, Ferrer-Sanchez 2024, Eq. 29).

Problem setup (physical window):
- x in [0, 1], discontinuity at x=0.5
- t_end = 0.4
- gamma = 5/3
- rho_L = 1.0, u_L = 0.5, p_L = 1.0
- rho_R = 1.0, u_R = -0.5, p_R = 1.0
"""

import sys
import importlib.util
from pathlib import Path
import numpy

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

from pysph.examples.gas_dynamics.sod_shocktube_rel import SodShockTubeRel, gamma, gamma1, dt
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


dim = 1
tf = 0.4


class RiemannProblem4Rel(SodShockTubeRel):
    def __init__(self, *args, **kwargs):
        super(RiemannProblem4Rel, self).__init__(*args, **kwargs)
        # Use a fine default for cleaner final publication-style curves.
        self.nl = 1000
        self.hdx = 1.0

    def initialize(self):
        # Computational domain is extended to reduce boundary-kernel artifacts.
        # Physical window for comparison remains x in [0, 1].
        self.xmin = -0.5
        self.xmax = 1.5
        self.plot_xmin = 0.0
        self.plot_xmax = 1.0
        self.x0 = 0.5
        # Problem 4 (reflection test, Eq. 29)
        self.rhol = 1.0
        self.rhor = 1.0
        self.pl = 1.0
        self.pr = 1.0
        self.ul = 0.5
        self.ur = -0.5

    def add_user_options(self, group):
        group.add_argument("--hdx", action="store", type=float, dest="hdx", default=1.0)
        group.add_argument("--nl", action="store", type=float, dest="nl", default=1000)
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
                "[DEBUG:resolution] using nl=%d, hdx=%.3f (Problem 4 reflection test)"
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
        rho0 = f.rho.copy()
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

        if self.rank == 0 and (not self._dbg_init_done):
            print("\n[DEBUG:init] Problem-4 target states (reflection):")
            print("  Left : rho=1.0, u=0.5,  p=1.0")
            print("  Right: rho=1.0, u=-0.5, p=1.0")
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

    def create_scheme(self):
        kf = getattr(self, "hdx", 1.0)
        rsolver = 0
        if self.rank == 0:
            print("[DEBUG:solver] Using SRHD Riemann solver = Rusanov (rsolver=%d)" % rsolver)
        return GSPHRelScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kf, rsolver=rsolver, niter=20, tol=1e-8
        )


if __name__ == '__main__':
    app = RiemannProblem4Rel()
    app.run()
    app.post_process()
