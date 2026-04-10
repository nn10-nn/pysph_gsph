"""Basic SRHD helper equations for GSPH."""

from compyle.api import declare
from math import sqrt
from pysph.sph.equation import Equation


class SRHDEOSFromConserved(Equation):
    """Recover primitive variables from (D, q, ehat) with Newton solve for chi.

    Here we store:
    - `rho` as lab-frame density D.
    - `qx,qy,qz` as conservative momentum components.
    - `ehat` as conservative energy-like variable.
    - `chi = p / D`.
    """
    def __init__(self, dest, sources, gamma=1.4, max_iter=40, tol=1e-10):
        super(SRHDEOSFromConserved, self).__init__(dest, sources)
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def loop(self, d_idx, d_rho, d_qx, d_qy, d_qz, d_ehat, d_chi,
             d_u, d_v, d_w, d_gamma_rel, d_hhat, d_rho_rest,
             d_p, d_e, d_cs):
        eps = 1e-14
        gm1 = self.gamma - 1.0

        qx = d_qx[d_idx]
        qy = d_qy[d_idx]
        qz = d_qz[d_idx]
        q2 = qx*qx + qy*qy + qz*qz
        ehat = d_ehat[d_idx]

        chi = d_chi[d_idx]
        if chi <= eps:
            # Warm start from previous pressure if needed.
            chi = max(eps, d_p[d_idx]/max(d_rho[d_idx], eps))

        it = declare('int')
        for it in range(self.max_iter):
            A = ehat + chi
            rad = A*A - q2
            if rad <= eps:
                chi = max(chi, sqrt(q2 + eps) - ehat + eps)
                A = ehat + chi
                rad = max(A*A - q2, eps)
            root = sqrt(rad)

            # f(chi) = q^2 + (ehat + chi)(chi/(g-1) - ehat) + sqrt((ehat+chi)^2-q^2)
            f = q2 + A * (chi/gm1 - ehat) + root
            df = (2.0*chi + (2.0 - self.gamma)*ehat)/gm1 + A/root
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

            if abs(alpha*dchi) < self.tol * max(1.0, abs(chi)):
                break

        A = ehat + chi
        rad = A*A - q2
        if rad <= eps:
            rad = eps
            A = sqrt(q2 + rad)
            chi = A - ehat

        invA = 1.0 / max(A, eps)
        ux = qx * invA
        vy = qy * invA
        wz = qz * invA
        vsq = ux*ux + vy*vy + wz*wz
        if vsq > 1.0 - 1e-12:
            scale = sqrt((1.0 - 1e-12)/vsq)
            ux *= scale
            vy *= scale
            wz *= scale
            vsq = ux*ux + vy*vy + wz*wz

        gamma_rel = 1.0/sqrt(max(1.0 - vsq, 1e-12))
        hhat = sqrt(rad)
        D = max(d_rho[d_idx], eps)
        p = chi * D
        rho_rest = D / gamma_rel
        eint = hhat - 1.0 - p/max(rho_rest, eps)
        if eint < 0.0:
            eint = 0.0
        cs2 = self.gamma * p / max(rho_rest*hhat, eps)
        if cs2 < 0.0:
            cs2 = 0.0
        if cs2 > 1.0 - 1e-12:
            cs2 = 1.0 - 1e-12

        d_chi[d_idx] = chi
        d_u[d_idx] = ux
        d_v[d_idx] = vy
        d_w[d_idx] = wz
        d_gamma_rel[d_idx] = gamma_rel
        d_hhat[d_idx] = hhat
        d_rho_rest[d_idx] = rho_rest
        d_p[d_idx] = p
        d_e[d_idx] = eint
        d_cs[d_idx] = sqrt(cs2)

