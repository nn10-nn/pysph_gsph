"""Integrator step for SRHD-GSPH."""

from math import sqrt
from pysph.sph.integrator_step import IntegratorStep


class GSPHRelStep(IntegratorStep):
    """One-step explicit update of (q, ehat), then primitive recovery."""

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               d_qx, d_qy, d_qz, d_ehat, d_chi, d_gamma_ad,
               d_gamma_rel, d_hhat, d_rho, d_rho_rest, d_p, d_e, d_cs,
               d_aqx, d_aqy, d_aqz, d_aeh, dt):
        eps = 1e-14

        # old velocity for position update
        u0 = d_u[d_idx]
        v0 = d_v[d_idx]
        w0 = d_w[d_idx]

        # conservative update
        qx = d_qx[d_idx] + dt*d_aqx[d_idx]
        qy = d_qy[d_idx] + dt*d_aqy[d_idx]
        qz = d_qz[d_idx] + dt*d_aqz[d_idx]
        ehat = d_ehat[d_idx] + dt*d_aeh[d_idx]

        g = d_gamma_ad[d_idx]
        gm1 = g - 1.0
        q2 = qx*qx + qy*qy + qz*qz
        chi = d_chi[d_idx]
        if chi <= eps:
            chi = max(eps, d_p[d_idx]/max(d_rho[d_idx], eps))

        # Newton solve for chi from conservative variables.
        for _ in range(40):
            A = ehat + chi
            rad = A*A - q2
            if rad <= eps:
                chi = max(chi, sqrt(q2 + eps) - ehat + eps)
                A = ehat + chi
                rad = max(A*A - q2, eps)
            root = sqrt(rad)
            f = q2 + A*(chi/gm1 - ehat) + root
            df = (2.0*chi + (2.0 - g)*ehat)/gm1 + A/root
            if abs(df) < eps:
                break
            dchi = -f/df
            alpha = 1.0
            while alpha > 1e-4:
                test = chi + alpha*dchi
                At = ehat + test
                if (test > eps) and ((At*At - q2) > eps):
                    break
                alpha *= 0.5
            chi = chi + alpha*dchi
            if abs(alpha*dchi) < 1e-10 * max(1.0, abs(chi)):
                break

        A = ehat + chi
        rad = A*A - q2
        if rad <= eps:
            rad = eps
            A = sqrt(q2 + rad)
            chi = A - ehat
        invA = 1.0/max(A, eps)

        un = qx*invA
        vn = qy*invA
        wn = qz*invA
        vsq = un*un + vn*vn + wn*wn
        if vsq > 1.0 - 1e-12:
            s = sqrt((1.0 - 1e-12)/vsq)
            un *= s
            vn *= s
            wn *= s
            vsq = un*un + vn*vn + wn*wn

        gamma_rel = 1.0/sqrt(max(1.0 - vsq, 1e-12))
        hhat = sqrt(rad)
        D = max(d_rho[d_idx], eps)
        p = chi * D
        rho_rest = D / gamma_rel
        eint = hhat - 1.0 - p/max(rho_rest, eps)
        if eint < 0.0:
            eint = 0.0
        cs2 = g * p / max(rho_rest*hhat, eps)
        if cs2 < 0.0:
            cs2 = 0.0
        if cs2 > 1.0 - 1e-12:
            cs2 = 1.0 - 1e-12

        d_qx[d_idx] = qx
        d_qy[d_idx] = qy
        d_qz[d_idx] = qz
        d_ehat[d_idx] = ehat
        d_chi[d_idx] = chi
        d_u[d_idx] = un
        d_v[d_idx] = vn
        d_w[d_idx] = wn
        d_gamma_rel[d_idx] = gamma_rel
        d_hhat[d_idx] = hhat
        d_rho_rest[d_idx] = rho_rest
        d_p[d_idx] = p
        d_e[d_idx] = eint
        d_cs[d_idx] = sqrt(cs2)

        # first-order position update with velocity averaging
        d_x[d_idx] += 0.5 * dt * (u0 + un)
        d_y[d_idx] += 0.5 * dt * (v0 + vn)
        d_z[d_idx] += 0.5 * dt * (w0 + wn)

