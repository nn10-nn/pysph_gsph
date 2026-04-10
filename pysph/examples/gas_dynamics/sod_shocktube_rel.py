"""1D SRHD Sod shock tube (paper setup) with standalone GSPHRelScheme."""

import os
import sys
from pathlib import Path
import numpy
from math import sqrt

from pysph.base.nnps import DomainManager
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
try:
    from pysph.sph.gas_dynamics.scheme_rel import GSPHRelScheme
except ModuleNotFoundError:
    # Fallback: force using local source tree instead of an older installed
    # pysph package when running this script directly.
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from pysph.sph.gas_dynamics.scheme_rel import GSPHRelScheme


dim = 1
gamma = 5.0/3.0
gamma1 = gamma - 1.0
dt = 1e-4
tf = 0.5


class SodShockTubeRel(ShockTubeSetup):
    def __init__(self, *args, **kwargs):
        super(SodShockTubeRel, self).__init__(*args, **kwargs)
        # Safe defaults so create_scheme() can be called before
        # consume_user_options().
        self.hdx = 1.2
        self.nl = 640
        self.dscheme = "constant_mass"

    def initialize(self):
        # Problem setup from the provided figure:
        # x in [0, 1], discontinuity at x=0.5, t in [0, 0.5].
        self.xmin = 0.0
        self.xmax = 1.0
        self.x0 = 0.5
        self.rhol = 1.0
        self.rhor = 0.125
        self.pl = 1.0
        self.pr = 0.1
        self.ul = 0.5
        self.ur = 0.5

    def add_user_options(self, group):
        group.add_argument("--hdx", action="store", type=float, dest="hdx", default=1.2)
        group.add_argument("--nl", action="store", type=float, dest="nl", default=640)
        group.add_argument(
            "--dscheme", choices=["constant_mass", "constant_volume"],
            dest="dscheme", default="constant_mass"
        )

    def consume_user_options(self):
        self.nl = int(self.options.nl)
        self.hdx = self.options.hdx
        self.dscheme = self.options.dscheme
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

        return [f]

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax, mirror_in_x=True, n_layers=2
        )

    def configure_scheme(self):
        self.scheme.configure_solver(tf=self.tf, dt=self.dt)

    def create_scheme(self):
        kf = getattr(self, "hdx", 1.2)
        return GSPHRelScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kf, rsolver=1, niter=20, tol=1e-8
        )

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

        x = pa.x
        D = pa.rho
        u = pa.u
        p = pa.p

        plt.plot(x, D, label='SRHD-GSPH')
        plt.xlabel('x')
        plt.ylabel('D')
        plt.legend()
        plt.xlim(self.xmin, self.xmax)
        plt.savefig(os.path.join(self.output_dir, "density_D.png"), dpi=300)
        plt.clf()

        plt.plot(x, u, label='SRHD-GSPH')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.xlim(self.xmin, self.xmax)
        plt.savefig(os.path.join(self.output_dir, "velocity.png"), dpi=300)
        plt.clf()

        plt.plot(x, p, label='SRHD-GSPH')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.legend()
        plt.xlim(self.xmin, self.xmax)
        plt.savefig(os.path.join(self.output_dir, "pressure.png"), dpi=300)
        plt.clf()

        fname = os.path.join(self.output_dir, 'results_rel.npz')
        numpy.savez(
            fname,
            # metadata
            t=self.tf, gamma=gamma, xmin=self.xmin, xmax=self.xmax, x0=self.x0,
            # primitive
            x=x, u=pa.u, v=pa.v, w=pa.w, p=pa.p, rho_rest=pa.rho_rest,
            e=pa.e, cs=pa.cs, hhat=pa.hhat, gamma_rel=pa.gamma_rel,
            # conservative
            D=pa.rho, qx=pa.qx, qy=pa.qy, qz=pa.qz, ehat=pa.ehat, chi=pa.chi
        )


if __name__ == '__main__':
    app = SodShockTubeRel()
    app.run()
    app.post_process()
