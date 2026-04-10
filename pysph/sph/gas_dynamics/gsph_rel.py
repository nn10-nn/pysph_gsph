"""SRHD-GSPH acceleration equation (first-order reconstruction)."""

from compyle.api import declare
from pysph.sph.equation import Equation


class GSPHAccelerationRel(Equation):
    def __init__(self, dest, sources, rsolver=1, gamma=1.4, niter=20,
                 tol=1e-6):
        super(GSPHAccelerationRel, self).__init__(dest, sources)
        self.rsolver = rsolver
        self.gamma = gamma
        self.niter = niter
        self.tol = tol

    def _get_helpers_(self):
        # Keep helper list empty to avoid codegen issues from helper parsing.
        return []

    def initialize(self, d_idx, d_aqx, d_aqy, d_aqz, d_aeh):
        d_aqx[d_idx] = 0.0
        d_aqy[d_idx] = 0.0
        d_aqz[d_idx] = 0.0
        d_aeh[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_p, d_chi, d_hhat, d_rho_rest, d_gamma_ad,
             d_u, d_v, d_w, d_qx, d_qy, d_qz, d_ehat,
             d_aqx, d_aqy, d_aqz, d_aeh,
             s_idx, s_m, s_rho, s_p, s_chi, s_hhat, s_rho_rest, s_gamma_ad,
             s_u, s_v, s_w, s_qx, s_qy, s_qz, s_ehat,
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

        # Conservative-to-primitive safety fallback for p, cs:
        # use p = chi * D if pressure is non-physical/undefined.
        p_l = s_p[s_idx]
        p_r = d_p[d_idx]
        if p_l <= 1e-14:
            p_l = max(s_chi[s_idx] * s_rho[s_idx], 1e-14)
        if p_r <= 1e-14:
            p_r = max(d_chi[d_idx] * d_rho[d_idx], 1e-14)

        cs_l = s_cs[s_idx]
        cs_r = d_cs[d_idx]
        if cs_l <= 1e-14:
            cs2_l = s_gamma_ad[s_idx] * p_l / max(
                s_rho_rest[s_idx] * s_hhat[s_idx], 1e-14
            )
            if cs2_l < 0.0:
                cs2_l = 0.0
            if cs2_l > 1.0 - 1e-12:
                cs2_l = 1.0 - 1e-12
            cs_l = cs2_l**0.5
        if cs_r <= 1e-14:
            cs2_r = d_gamma_ad[d_idx] * p_r / max(
                d_rho_rest[d_idx] * d_hhat[d_idx], 1e-14
            )
            if cs2_r < 0.0:
                cs2_r = 0.0
            if cs2_r > 1.0 - 1e-12:
                cs2_r = 1.0 - 1e-12
            cs_r = cs2_r**0.5

        # Local relativistic characteristic speeds.
        uc_l = u_l * cs_l
        uc_r = u_r * cs_r
        den_lp = 1.0 + uc_l
        den_lm = 1.0 - uc_l
        den_rp = 1.0 + uc_r
        den_rm = 1.0 - uc_r
        if abs(den_lp) < 1e-14:
            den_lp = 1e-14 if den_lp >= 0.0 else -1e-14
        if abs(den_lm) < 1e-14:
            den_lm = 1e-14 if den_lm >= 0.0 else -1e-14
        if abs(den_rp) < 1e-14:
            den_rp = 1e-14 if den_rp >= 0.0 else -1e-14
        if abs(den_rm) < 1e-14:
            den_rm = 1e-14 if den_rm >= 0.0 else -1e-14

        lam_lm = (u_l - cs_l) / den_lm
        lam_lp = (u_l + cs_l) / den_lp
        lam_rm = (u_r - cs_r) / den_rm
        lam_rp = (u_r + cs_r) / den_rp

        # Rusanov (0) or HLL/HLLC(fallback-to-HLL).
        if self.rsolver == 0:
            lam_max = abs(lam_lm)
            if abs(lam_lp) > lam_max:
                lam_max = abs(lam_lp)
            if abs(lam_rm) > lam_max:
                lam_max = abs(lam_rm)
            if abs(lam_rp) > lam_max:
                lam_max = abs(lam_rp)

            pstar = 0.5 * (p_l + p_r) - 0.5 * lam_max * (q_r - q_l)
            if pstar <= 1e-14:
                pstar = 1e-14
            num = 0.5 * (p_l * u_l + p_r * u_r) - 0.5 * lam_max * (
                d_ehat[d_idx] - s_ehat[s_idx]
            )
            ustar = num / pstar
        else:
            # HLL (used for HLL and HLLC in this robust implementation).
            s_l = lam_lm if lam_lm < lam_rm else lam_rm
            s_r = lam_lp if lam_lp > lam_rp else lam_rp
            den = s_r - s_l
            if abs(den) < 1e-14:
                den = 1e-14 if den >= 0.0 else -1e-14

            pstar = (s_r * p_l - s_l * p_r + s_l * s_r * (q_r - q_l)) / den
            if pstar <= 1e-14:
                pstar = 1e-14
            num = (s_r * p_l * u_l - s_l * p_r * u_r + s_l * s_r * (
                d_ehat[d_idx] - s_ehat[s_idx]
            )) / den
            ustar = num / pstar

        if ustar > 0.999999:
            ustar = 0.999999
        elif ustar < -0.999999:
            ustar = -0.999999

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
