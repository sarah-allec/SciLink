"""Live DFT parameter-convergence test (real VASP, run in a SLURM allocation).

Validates the engine-specific + real-VASP pieces that unit tests can't reach:
the VASP param-setter producing valid INCAR edits, real ladder execution, and
`read_convergence_observable` reading energies from real vasprun.xml. It drives
the convergence machinery directly (`converge_parameters`) on a small cell, so
it does not depend on POTCAR-in-pipeline or the full workflow.

Run inside a compute-node allocation (VASP runs in-process via LocalExecutor):

    export VASP_PP_PATH=/people/alle927/VASP_POT/potpaw_PBE   # per-element POTCAR root
    export VASP_RUN_CMD="mpirun vasp_std"                     # + module loads (see sbatch)
    python tests/test_dft_convergence_live.py --system cu

`--system cu` (FCC metal, k-point-sensitive) or `--system si` (diamond
semiconductor). The energy/atom ladders should each show a clear plateau.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_ROOT = REPO_ROOT / "tests" / "_dft_convergence_runs"

# Minimal single-point INCAR per system type. The convergence observable is the
# energy/atom, so these are static SCF (IBRION=-1, NSW=0). KSPACING lives in the
# INCAR (no KPOINTS file) so the k-point ladder can vary it.
_INCAR = {
    "cu": (  # FCC metal — Methfessel-Paxton smearing
        "SYSTEM = Cu convergence\nPREC = Accurate\nENCUT = 400\n"
        "ISMEAR = 1\nSIGMA = 0.2\nEDIFF = 1E-6\nLREAL = .FALSE.\n"
        "IBRION = -1\nNSW = 0\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n"
        "KSPACING = 0.3\n"
    ),
    "si": (  # diamond semiconductor — Gaussian smearing
        "SYSTEM = Si convergence\nPREC = Accurate\nENCUT = 400\n"
        "ISMEAR = 0\nSIGMA = 0.05\nEDIFF = 1E-6\nLREAL = .FALSE.\n"
        "IBRION = -1\nNSW = 0\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n"
        "KSPACING = 0.3\n"
    ),
}


def _build_structure(system: str, poscar_path: Path):
    from ase.build import bulk
    from ase.io import write
    if system == "cu":
        atoms = bulk("Cu", "fcc", a=3.61)
    elif system == "si":
        atoms = bulk("Si", "diamond", a=5.43)
    else:
        raise SystemExit(f"unknown --system {system!r}")
    write(str(poscar_path), atoms, format="vasp", sort=True, direct=True)


def _species_from_poscar(poscar_path: Path):
    """VASP5 POSCAR line 6 lists the element symbols in file order."""
    lines = poscar_path.read_text().splitlines()
    return lines[5].split()


def _assemble_potcar(species, pp_root: Path) -> str:
    parts = []
    for el in species:
        p = pp_root / el / "POTCAR"
        if not p.is_file():
            raise SystemExit(f"missing POTCAR for {el} at {p} "
                             f"(is VASP_PP_PATH the per-element root?)")
        parts.append(p.read_text())
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["cu", "si"], default="cu")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    pp_root = os.environ.get("VASP_PP_PATH")
    run_cmd = os.environ.get("VASP_RUN_CMD", "mpirun vasp_std")
    if not pp_root:
        raise SystemExit("set VASP_PP_PATH to the per-element POTCAR root")
    pp_root = Path(pp_root)

    work = RUN_ROOT / args.system
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    poscar = work / "POSCAR"
    _build_structure(args.system, poscar)
    species = _species_from_poscar(poscar)
    print(f"system={args.system}  species={species}  run_cmd={run_cmd!r}")

    base_inputs = {
        "INCAR": _INCAR[args.system],
        "POSCAR": poscar.read_text(),
        "POTCAR": _assemble_potcar(species, pp_root),
    }

    from scilink.skills.loader import load_skill
    from scilink.skills._shared._registry import get_tool_function
    from scilink.agents.sim_agents.convergence import converge_parameters
    from scilink.agents.sim_agents.refinement import LocalExecutor

    specs = load_skill("vasp", domain="periodic_dft")["meta"]["convergence"]
    print("convergence specs:", json.dumps(specs))

    from scilink.skills.periodic_dft.vasp.vasp_convergence import kspacing_to_mesh

    set_param = get_tool_function("set_convergence_param", active_skills=["vasp"])
    read_obs = get_tool_function("read_convergence_observable", active_skills=["vasp"])
    executor = LocalExecutor(timeout=args.timeout)

    def run_ladder(param, members):
        dirs = {}
        root = work / "convergence" / str(param)
        for setting, inputs in members.items():
            rdir = root / str(setting)
            rdir.mkdir(parents=True, exist_ok=True)
            print(f"  [{param}={setting}] running VASP in {rdir} ...")
            res = executor.run(inputs, run_cmd, str(rdir))
            print(f"    -> {res.get('status')} rc={res.get('returncode')}")
            dirs[setting] = str(rdir)
        return dirs

    pc = converge_parameters(
        base_inputs=base_inputs, specs=specs,
        set_param=lambda i, p, v: set_param(input_files=i, param=p, value=v),
        read_observable=lambda d, o: read_obs(output_dir=d, observable=o),
        run_ladder=run_ladder,
    )

    print("\n" + "=" * 60 + "\nRESULTS\n" + "=" * 60)
    print(f"all_converged: {pc.all_converged}")
    ok = True
    for s in pc.sweeps:
        print(f"\n{s.param_name}:")
        for setting, value in s.observations:
            run_dir = s.run_dirs.get(setting)
            mesh = kspacing_to_mesh(run_dir) if run_dir else None
            label = f"KSPACING {setting:>5} ({mesh})" if mesh else f"{setting:>8}"
            print(f"  {label}: {value}")
        c = s.convergence
        adopted_mesh = (kspacing_to_mesh(s.run_dirs.get(c.setting))
                        if c.converged and c.setting in s.run_dirs else None)
        setting_label = (f"{c.setting} ({adopted_mesh})" if adopted_mesh
                         else c.setting)
        print(f"  -> converged={c.converged} setting={setting_label} "
              f"value={c.value}\n     {c.reason}")
        # Every rung must have produced a readable energy — that is the real-VASP
        # extraction check. Convergence itself is reported, not required.
        if any(v is None for _, v in s.observations):
            print("  !! a rung produced no readable energy — VASP or extraction "
                  "failed there")
            ok = False

    print("\n" + ("PASS: every rung produced an energy" if ok
                  else "FAIL: some rung produced no energy — see above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
