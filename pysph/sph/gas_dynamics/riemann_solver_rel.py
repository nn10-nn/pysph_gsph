"""Approximate Riemann solvers for SRHD-GSPH (Lagrangian form)."""

from math import sqrt


RUSANOV = 0
HLL = 1
HLLC = 2


def _safe(v=0.0, floor=1e-14):
    if v >= 0.0:
        return v if v > floor else floor
    return -((-v) if (-v) > floor else floor)


def _rel_char_speeds(u=0.0, cs=0.0, result=[0.0, 0.0]):
    # Relativistic Lagrangian eigenvalues:
    # lambda_1 = -cs / (Gamma * (1 - u*cs))
    # lambda_3 =  cs / (Gamma * (1 + u*cs))
    uc = u * cs
    den_p = _safe(1.0 + uc)
    den_m = _safe(1.0 - uc)
    gm = 1.0 / ((1.0 - u*u) if (1.0 - u*u) > 1e-12 else 1e-12) ** 0.5
    result[0] = -cs / (gm * den_m)
    result[1] = cs / (gm * den_p)


def riemann_solve_rel(
    method=1, rho_l=1.0, rho_r=1.0, p_l=1.0, p_r=1.0, u_l=0.0, u_r=0.0,
    cs_l=0.0, cs_r=0.0, q_l=0.0, q_r=0.0, ehat_l=0.0, ehat_r=0.0,
    gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]
):
    if method == RUSANOV:
        return rusanov_rel(
            rho_l, rho_r, p_l, p_r, u_l, u_r, cs_l, cs_r,
            q_l, q_r, ehat_l, ehat_r, gamma, niter, tol, result
        )
    elif method == HLLC:
        # Use HLL as robust fallback in this implementation.
        return hll_rel(
            rho_l, rho_r, p_l, p_r, u_l, u_r, cs_l, cs_r,
            q_l, q_r, ehat_l, ehat_r, gamma, niter, tol, result
        )
    else:
        return hll_rel(
            rho_l, rho_r, p_l, p_r, u_l, u_r, cs_l, cs_r,
            q_l, q_r, ehat_l, ehat_r, gamma, niter, tol, result
        )


def rusanov_rel(
    rho_l=1.0, rho_r=1.0, p_l=1.0, p_r=1.0, u_l=0.0, u_r=0.0, cs_l=0.0,
    cs_r=0.0, q_l=0.0, q_r=0.0, ehat_l=0.0, ehat_r=0.0, gamma=1.4, niter=20,
    tol=1e-6, result=[0.0, 0.0]
):
    lam_l = [0.0, 0.0]
    lam_r = [0.0, 0.0]
    _rel_char_speeds(u_l, cs_l, lam_l)
    _rel_char_speeds(u_r, cs_r, lam_r)
    lam_max = max(abs(lam_l[0]), abs(lam_l[1]), abs(lam_r[0]), abs(lam_r[1]))

    pstar = 0.5 * (p_l + p_r) - 0.5 * lam_max * (q_r - q_l)
    if pstar <= 1e-14:
        pstar = 1e-14

    num = 0.5 * (p_l * u_l + p_r * u_r) - 0.5 * lam_max * (ehat_r - ehat_l)
    ustar = num / pstar
    if ustar > 0.999999:
        ustar = 0.999999
    elif ustar < -0.999999:
        ustar = -0.999999

    result[0] = pstar
    result[1] = ustar
    return 0


def hll_rel(
    rho_l=1.0, rho_r=1.0, p_l=1.0, p_r=1.0, u_l=0.0, u_r=0.0, cs_l=0.0,
    cs_r=0.0, q_l=0.0, q_r=0.0, ehat_l=0.0, ehat_r=0.0, gamma=1.4, niter=20,
    tol=1e-6, result=[0.0, 0.0]
):
    lam_l = [0.0, 0.0]
    lam_r = [0.0, 0.0]
    _rel_char_speeds(u_l, cs_l, lam_l)
    _rel_char_speeds(u_r, cs_r, lam_r)

    s_l = min(lam_l[0], lam_r[0])
    s_r = max(lam_l[1], lam_r[1])
    den = s_r - s_l
    if abs(den) < 1e-14:
        return rusanov_rel(
            rho_l, rho_r, p_l, p_r, u_l, u_r, cs_l, cs_r,
            q_l, q_r, ehat_l, ehat_r, gamma, niter, tol, result
        )

    pstar = (s_r * p_l - s_l * p_r + s_l * s_r * (q_r - q_l)) / den
    if pstar <= 1e-14:
        pstar = 1e-14

    num = (s_r * p_l * u_l - s_l * p_r * u_r +
           s_l * s_r * (ehat_r - ehat_l)) / den
    ustar = num / pstar
    if ustar > 0.999999:
        ustar = 0.999999
    elif ustar < -0.999999:
        ustar = -0.999999

    result[0] = pstar
    result[1] = ustar
    return 0


HELPERS_REL = [
    _safe, _rel_char_speeds, riemann_solve_rel, rusanov_rel, hll_rel
]
