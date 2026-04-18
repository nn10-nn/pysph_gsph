"""Case definitions for the three 1D SRHD benchmarks used in Chapter 5."""

from __future__ import annotations


CASES = {
    "case1": {
        "label": "Case 1",
        "desc": "Standard shock-tube",
        "rho_l": 1.0,
        "u_l": 0.5,
        "p_l": 1.0,
        "rho_r": 0.125,
        "u_r": 0.5,
        "p_r": 0.1,
        "gamma": 5.0 / 3.0,
        "t": 0.5,
        "x1": 0.0,
        "x2": 1.0,
        "num_npz_rel": "sod_shocktube_rel_output/results_rel.npz",
        "fig_prefix": "fig5-1-case1-1d",
    },
    "case2": {
        "label": "Case 2",
        "desc": "Mixed wave-pattern problem",
        "rho_l": 1.0,
        "u_l": 0.5,
        "p_l": 0.1,
        "rho_r": 1.0,
        "u_r": 0.0,
        "p_r": 1.0,
        "gamma": 5.0 / 3.0,
        "t": 0.4,
        "x1": 0.0,
        "x2": 1.0,
        "num_npz_rel": "riemann_problem2_rel_output/results_rel.npz",
        "fig_prefix": "fig5-2-case2-1d",
    },
    "case3": {
        "label": "Case 3",
        "desc": "Reflection/collision problem",
        "rho_l": 1.0,
        "u_l": 0.5,
        "p_l": 1.0,
        "rho_r": 1.0,
        "u_r": -0.5,
        "p_r": 1.0,
        "gamma": 5.0 / 3.0,
        "t": 0.4,
        "x1": 0.0,
        "x2": 1.0,
        "num_npz_rel": "riemann_problem4_rel_output/results_rel.npz",
        "fig_prefix": "fig5-3-case3-1d",
    },
}

