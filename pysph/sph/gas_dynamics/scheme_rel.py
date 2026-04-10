"""Standalone SRHD-GSPH scheme (keeps classical schemes untouched)."""

from pysph.sph.scheme import Scheme, add_bool_argument


class GSPHRelScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma=1.4, kernel_factor=1.2,
                 rsolver=1, niter=20, tol=1e-6):
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.kernel_factor = kernel_factor
        self.rsolver = rsolver
        self.niter = niter
        self.tol = tol
        self.rsolver_choices = {
            'rusanov': 0,
            'hll': 1,
            'hllc': 2,
        }

    def add_user_options(self, group):
        group.add_argument(
            "--rsolver-rel", action="store", type=str, dest="rsolver_rel",
            default=None, choices=set(self.rsolver_choices.keys()),
            help=f"Relativistic Riemann solver, one of {set(self.rsolver_choices.keys())}"
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None, help="Gamma for EOS."
        )
        group.add_argument(
            "--kernel-factor", action="store", type=float, dest="kernel_factor",
            default=None, help="Kernel factor k in h = k (m/D)^(1/d)."
        )
        add_bool_argument(
            group, "dummy-rel-flag", dest="dummy_rel_flag",
            help="Reserved flag for future SRHD options.", default=False
        )

    def consume_user_options(self, options):
        data = {
            'gamma': self._smart_getattr(options, 'gamma'),
            'kernel_factor': self._smart_getattr(options, 'kernel_factor'),
            'rsolver': self._smart_getattr_mapped(options, 'rsolver_rel'),
        }
        self.configure(**data)

    def _smart_getattr_mapped(self, obj, var):
        res = getattr(obj, var)
        if res is None:
            return self.rsolver
        return self.rsolver_choices[res]

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import Gaussian
        from pysph.sph.integrator import EulerIntegrator
        from pysph.solver.solver import Solver
        from pysph.sph.gas_dynamics.integrator_step_rel import GSPHRelStep

        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = GSPHRelStep()

        cls = integrator_cls if integrator_cls is not None else EulerIntegrator
        integrator = cls(**steppers)
        self.solver = Solver(dim=self.dim, integrator=integrator, kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.gas_dynamics.basic import (
            ScaleSmoothingLength, SummationDensity, UpdateSmoothingLengthFromVolume
        )
        from pysph.sph.gas_dynamics.basic_rel import SRHDEOSFromConserved
        from pysph.sph.gas_dynamics.gsph_rel import GSPHAccelerationRel

        equations = []
        all_pa = self.fluids + self.solids

        g0 = [ScaleSmoothingLength(dest=f, sources=None, factor=2.0)
              for f in self.fluids]
        equations.append(Group(equations=g0, update_nnps=True))

        g1 = [SummationDensity(dest=f, sources=all_pa, dim=self.dim)
              for f in self.fluids]
        equations.append(Group(equations=g1, update_nnps=False))

        g2 = [UpdateSmoothingLengthFromVolume(
            dest=f, sources=None, k=self.kernel_factor, dim=self.dim
        ) for f in self.fluids]
        equations.append(Group(equations=g2, update_nnps=True))

        g3 = [SummationDensity(dest=f, sources=all_pa, dim=self.dim)
              for f in self.fluids]
        equations.append(Group(equations=g3, update_nnps=False))

        g4 = [SRHDEOSFromConserved(dest=f, sources=None, gamma=self.gamma)
              for f in self.fluids]
        equations.append(Group(equations=g4, update_nnps=False))

        g5 = [GSPHAccelerationRel(
            dest=f, sources=all_pa, rsolver=self.rsolver, gamma=self.gamma,
            niter=self.niter, tol=self.tol
        ) for f in self.fluids]
        equations.append(Group(equations=g5, update_nnps=False))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_gasd
        import numpy

        particle_arrays = {p.name: p for p in particles}
        dummy = get_particle_array_gasd(name='junk')
        base_props = list(dummy.properties.keys())
        extra_props = [
            'qx', 'qy', 'qz', 'aqx', 'aqy', 'aqz',
            'ehat', 'aeh', 'chi', 'gamma_rel', 'hhat', 'rho_rest',
            'gamma_ad'
        ]
        props = base_props + extra_props

        output_props = [
            'x', 'y', 'z', 'u', 'v', 'w',
            'rho', 'rho_rest', 'p', 'e', 'cs',
            'qx', 'qy', 'qz', 'ehat', 'chi',
            'gamma_rel', 'hhat', 'gamma_ad',
            'aqx', 'aqy', 'aqz', 'aeh',
            'm', 'h'
        ]

        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.gamma_ad[:] = self.gamma
            pa.set_output_arrays(output_props)

        solid_props = set(props) | {'wij', 'htmp'}
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.gamma_ad[:] = self.gamma
            pa.set_output_arrays(output_props)
