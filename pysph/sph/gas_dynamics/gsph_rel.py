"""SRHD-GSPH acceleration equation (first-order reconstruction)."""

from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.sph.gas_dynamics.riemann_solver_rel import (
    HELPERS_REL, riemann_solve_rel, RUSANOV, HLL, HLLC
)


class GSPHAccelerationRel(Equation):
    def __init__(self, dest, sources, rsolver=HLL, gamma=1.4, niter=20,
                 tol=1e-6):
        super(GSPHAccelerationRel, self).__init__(dest, sources)
        self.rsolver = rsolver
        self.gamma = gamma
        self.niter = niter
        self.tol = tol

    def _get_helpers_(self):
        return HELPERS_REL

    def initialize(self, d_idx, d_aqx, d_aqy, d_aqz, d_aeh):
        d_aqx[d_idx] = 0.0
        d_aqy[d_idx] = 0.0
        d_aqz[d_idx] = 0.0
        d_aeh[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_p, d_u, d_v, d_w, d_qx, d_qy, d_qz, d_ehat,
             d_aqx, d_aqy, d_aqz, d_aeh,
             s_idx, s_m, s_rho, s_p, s_u, s_v, s_w, s_qx, s_qy, s_qz, s_ehat,
             d_cs, s_cs,
             XIJ, RIJ, EPS, DWI, DWJ):
        eij = declare('matrix(3)')
        if RIJ < 1e-14:
            eij[0] = 0.0
            eij[1] = 0.0
            eij[2] = 0.0
        else:
            inv = 1.0/RIJ
            eij[0] = XIJ[0] * inv
            eij[1] = XIJ[1] * inv
            eij[2] = XIJ[2] * inv

        # Local 1D Riemann states along particle-pair normal.
        u_l = s_u[s_idx]*eij[0] + s_v[s_idx]*eij[1] + s_w[s_idx]*eij[2]
        u_r = d_u[d_idx]*eij[0] + d_v[d_idx]*eij[1] + d_w[d_idx]*eij[2]
        q_l = s_qx[s_idx]*eij[0] + s_qy[s_idx]*eij[1] + s_qz[s_idx]*eij[2]
        q_r = d_qx[d_idx]*eij[0] + d_qy[d_idx]*eij[1] + d_qz[d_idx]*eij[2]

        result = declare('matrix(2)')
        riemann_solve_rel(
            self.rsolver,
            s_rho[s_idx], d_rho[d_idx],
            s_p[s_idx], d_p[d_idx],
            u_l, u_r, s_cs[s_idx], d_cs[d_idx], q_l, q_r,
            s_ehat[s_idx], d_ehat[d_idx], self.gamma, self.niter, self.tol,
            result
        )

        # Use pressure from particle arrays after EOS recovery.
        # (Done this way to keep Riemann call signature explicit and stable.)
        pstar = result[0]
        ustar = result[1]

        # 3D interface velocity: normal star velocity + averaged tangential.
        vstar = declare('matrix(3)')
        avg_u = 0.5 * (d_u[d_idx] + s_u[s_idx])
        avg_v = 0.5 * (d_v[d_idx] + s_v[s_idx])
        avg_w = 0.5 * (d_w[d_idx] + s_w[s_idx])
        avg_n = 0.5 * (u_l + u_r)
        vstar[0] = ustar*eij[0] + (avg_u - avg_n*eij[0])
        vstar[1] = ustar*eij[1] + (avg_v - avg_n*eij[1])
        vstar[2] = ustar*eij[2] + (avg_w - avg_n*eij[2])

        rhoi = max(d_rho[d_idx], 1e-14)
        rhoj = max(s_rho[s_idx], 1e-14)
        gx = (1.0/(rhoi*rhoi))*DWI[0] + (1.0/(rhoj*rhoj))*DWJ[0]
        gy = (1.0/(rhoi*rhoi))*DWI[1] + (1.0/(rhoj*rhoj))*DWJ[1]
        gz = (1.0/(rhoi*rhoi))*DWI[2] + (1.0/(rhoj*rhoj))*DWJ[2]

        mj = s_m[s_idx]
        d_aqx[d_idx] += -mj * pstar * gx
        d_aqy[d_idx] += -mj * pstar * gy
        d_aqz[d_idx] += -mj * pstar * gz

        d_aeh[d_idx] += -mj * pstar * (vstar[0]*gx + vstar[1]*gy + vstar[2]*gz)
