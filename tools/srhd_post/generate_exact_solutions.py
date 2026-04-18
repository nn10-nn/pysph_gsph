"""Generate exact/reference SRHD solutions via Exact_Riemann_Solver (RMHD -> SRHD degenerate setup).

RMHD is degenerated to SRHD by forcing:
  Bx=By=Bz=0 and vy=vz=0.
Then solution.dat columns are read as:
  x, rho, p, vx, vy, vz, By, Bz
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

try:
    from tools.srhd_post.case_definitions import CASES
except ModuleNotFoundError:
    # Allow direct execution:
    #   python tools/srhd_post/generate_exact_solutions.py
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.srhd_post.case_definitions import CASES


def _write_rinput(path: Path, case: Dict[str, float]) -> None:
    # initialdata.f90 reads each line with format(a22, f40.20)
    entries = [
        ("Bx", 0.0),
        ("gamma", case["gamma"]),
        ("rho_left", case["rho_l"]),
        ("pgas_left", case["p_l"]),
        ("vx_left", case["u_l"]),
        ("vy_left", 0.0),
        ("vz_left", 0.0),
        ("By_left", 0.0),
        ("Bz_left", 0.0),
        ("rho_right", case["rho_r"]),
        ("pgas_right", case["p_r"]),
        ("vx_right", case["u_r"]),
        ("vy_right", 0.0),
        ("vz_right", 0.0),
        ("By_right", 0.0),
        ("Bz_right", 0.0),
    ]
    lines = [f"{k:<22}{v:40.20f}\n" for (k, v) in entries]
    path.write_text("".join(lines), encoding="utf-8")


def _find_existing_solver_dir(preferred: Path) -> Path:
    """Find a valid Exact_Riemann_Solver directory."""
    candidates = [
        preferred,
        Path("third_party/Exact_Riemann_Solver"),
        Path("Exact_Riemann_Solver"),
    ]
    # Also try relative to repo root when script is run from other directories.
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    candidates.extend([
        repo_root / "third_party" / "Exact_Riemann_Solver",
        repo_root / "Exact_Riemann_Solver",
    ])

    for c in candidates:
        p = c.resolve()
        if not p.exists():
            continue
        # Minimal validity check: executable OR known source file.
        if (p / "riemann_rmhd").exists() or (p / "riemann_rmhd.f90").exists():
            return p
    raise FileNotFoundError(
        "Could not locate Exact_Riemann_Solver directory.\n"
        f"Tried candidates:\n  - " + "\n  - ".join(str(x.resolve()) for x in candidates) + "\n"
        "Please clone it, e.g.:\n"
        "  git clone https://github.com/bgiacoma/Exact_Riemann_Solver third_party/Exact_Riemann_Solver"
    )


def _compile_with_gfortran(solver_dir: Path) -> None:
    srcs = [
        "Interfaces.f90",
        "initialdata.f90",
        "quartic.f90",
        "nrutil.f90",
        "lubksb.f90",
        "ludcmp.f90",
        "postshock.f90",
        "riemann_rmhd.f90",
    ]
    missing = [s for s in srcs if not (solver_dir / s).exists()]
    if missing:
        raise RuntimeError(
            "Cannot fallback-compile with gfortran, missing source files:\n  - "
            + "\n  - ".join(missing)
        )
    cmd = ["gfortran", "-ffree-line-length-none", *srcs, "-o", "riemann_rmhd"]
    proc = subprocess.run(
        cmd,
        cwd=str(solver_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            "gfortran fallback build failed.\n"
            f"Command: {' '.join(cmd)}\n"
            "Please ensure gfortran is installed and callable in PATH."
        )


def _ensure_solver_built(solver_dir: Path) -> Path:
    exe = solver_dir / "riemann_rmhd"
    if exe.exists():
        return exe

    # Build path diagnosis.
    has_makefile = any((solver_dir / n).exists() for n in ("Makefile", "makefile", "GNUmakefile"))
    print(f"[step] building Exact_Riemann_Solver in: {solver_dir}")
    if has_makefile:
        proc = subprocess.run(
            ["make"],
            cwd=str(solver_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to build Exact_Riemann_Solver with make.\n"
                f"solver_dir={solver_dir}\n"
                "Check whether gfortran is installed and whether Makefile targets are valid."
            )
    else:
        print("[warn] Makefile not found, trying direct gfortran fallback compilation...")
        _compile_with_gfortran(solver_dir)

    if not exe.exists():
        raise RuntimeError(
            "Build finished but executable 'riemann_rmhd' was not produced.\n"
            f"solver_dir={solver_dir}"
        )
    return exe


def _run_case(
    exe: Path,
    solver_dir: Path,
    out_dir: Path,
    case_key: str,
    case: Dict[str, float],
    nx_ref: int,
    accuracy: float,
    verbose: int,
) -> Path:
    # Prepare RInput.txt for user-defined initial condition (initial_data = 0).
    rinput_path = solver_dir / "RInput.txt"
    _write_rinput(rinput_path, case)

    # Interactive input for riemann_rmhd:
    # 0=user-defined IC, 1=ideal EOS, x1, x2, t, nx, accuracy, verbosity
    input_blob = (
        "0\n"
        "1\n"
        f"{case['x1']}\n"
        f"{case['x2']}\n"
        f"{case['t']}\n"
        f"{nx_ref}\n"
        f"{accuracy}\n"
        f"{verbose}\n"
    )
    proc = subprocess.run(
        [str(exe)],
        cwd=str(solver_dir),
        input=input_blob,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_file = out_dir / f"{case_key}_solver.log"
    log_file.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Exact solver failed for {case_key}. See {log_file} for details."
        )

    sol_path = solver_dir / "solution.dat"
    if not sol_path.exists():
        raise RuntimeError(
            f"Expected {sol_path} not found after running exact solver for {case_key}."
        )
    data = np.loadtxt(sol_path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 8:
        raise RuntimeError(
            f"Unexpected solution.dat format for {case_key}: shape={data.shape}"
        )

    x = data[:, 0]
    rho = data[:, 1]
    p = data[:, 2]
    vx = data[:, 3]
    vy = data[:, 4]
    vz = data[:, 5]
    by = data[:, 6]
    bz = data[:, 7]

    idx = np.argsort(x)
    x = x[idx]
    rho = rho[idx]
    p = p[idx]
    vx = vx[idx]
    vy = vy[idx]
    vz = vz[idx]
    by = by[idx]
    bz = bz[idx]

    out_npz = out_dir / f"{case_key}_exact.npz"
    np.savez(
        out_npz,
        case_key=case_key,
        t=case["t"],
        gamma=case["gamma"],
        x=x,
        rho=rho,
        p=p,
        u=vx,
        vx=vx,
        vy=vy,
        vz=vz,
        by=by,
        bz=bz,
    )
    # Keep raw artifacts for traceability/reproducibility.
    shutil.copy2(sol_path, out_dir / f"{case_key}_solution.dat")
    exact_sol = solver_dir / "exact.sol"
    if exact_sol.exists():
        shutil.copy2(exact_sol, out_dir / f"{case_key}_exact.sol")
    print(f"[ok] {case_key}: saved exact solution -> {out_npz}")
    return out_npz


def _iter_selected_cases(case_keys: Iterable[str]):
    for key in case_keys:
        if key not in CASES:
            raise KeyError(f"Unknown case '{key}'. Valid keys: {list(CASES.keys())}")
        yield key, CASES[key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact/reference solutions for three 1D SRHD cases "
                    "using Exact_Riemann_Solver with RMHD->SRHD degeneration."
    )
    parser.add_argument(
        "--solver-dir",
        type=Path,
        default=Path("third_party/Exact_Riemann_Solver"),
        help="Path to cloned Exact_Riemann_Solver repository.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("postprocess/exact_riemann"),
        help="Directory to store exact/reference outputs.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["case1", "case2", "case3"],
        help="Subset of cases to run. Default: case1 case2 case3",
    )
    parser.add_argument(
        "--nx-ref",
        type=int,
        default=4000,
        help="Number of plotting grid points requested to exact solver.",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=1.0e-10,
        help="Exact solver accuracy parameter.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Exact solver verbosity: 0 minimal, 1 normal, 2 detailed.",
    )
    args = parser.parse_args()

    solver_dir = _find_existing_solver_dir(args.solver_dir)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step] using solver_dir: {solver_dir}")
    exe = _ensure_solver_built(solver_dir)

    print("[step] generating exact/reference solutions...")
    for key, case in _iter_selected_cases(args.cases):
        print(
            f"[case] {key}: t={case['t']}, gamma={case['gamma']}, "
            f"L=({case['rho_l']},{case['u_l']},{case['p_l']}), "
            f"R=({case['rho_r']},{case['u_r']},{case['p_r']})"
        )
        _run_case(
            exe=exe,
            solver_dir=solver_dir,
            out_dir=out_dir,
            case_key=key,
            case=case,
            nx_ref=args.nx_ref,
            accuracy=args.accuracy,
            verbose=args.verbose,
        )
    print("[done] exact/reference generation finished.")


if __name__ == "__main__":
    main()
