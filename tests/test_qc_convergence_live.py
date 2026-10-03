"""Live QC basis-convergence test (real NWChem, run in a SLURM allocation).

Validates the NWChem pieces unit tests can't reach: the basis param-setter
producing a runnable deck, real ladder execution, and
`read_convergence_observable` reading the total energy (via cclib) from real
NWChem output. Drives `converge_parameters` directly on a small molecule.

Run inside an allocation (NWChem runs in-process via LocalExecutor):

    export NWCHEM_RUN_CMD="mpirun nwchem job.nw"   # + module loads (see sbatch)
    python tests/test_qc_convergence_live.py
"""

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_ROOT = REPO_ROOT / "tests" / "_qc_convergence_runs"

# Minimal single-point B3LYP water deck; the basis block is what the sweep edits.
_DECK = (
    "start conv\n"
    "geometry units angstrom\n"
    "  O  0.00000  0.00000  0.11926\n"
    "  H  0.00000  0.76324 -0.47704\n"
    "  H  0.00000 -0.76324 -0.47704\n"
    "end\n"
    "basis spherical\n"
    "  * library def2-svp\n"
    "end\n"
    "dft\n"
    "  xc b3lyp\n"
    "  mult 1\n"
    "end\n"
    "task dft energy\n"
)


def main():
    run_cmd = os.environ.get("NWCHEM_RUN_CMD", "nwchem job.nw")
    work = RUN_ROOT
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    print(f"run_cmd={run_cmd!r}")

    from scilink.skills.loader import load_skill
    from scilink.skills._shared._registry import get_tool_function
    from scilink.agents.sim_agents.convergence import converge_parameters
    from scilink.agents.sim_agents.refinement import LocalExecutor

    specs = load_skill("nwchem", domain="molecular_qc")["meta"]["convergence"]
    print("convergence specs:", specs)

    set_param = get_tool_function("set_convergence_param", active_skills=["nwchem"])
    read_obs = get_tool_function("read_convergence_observable", active_skills=["nwchem"])
    executor = LocalExecutor(timeout=1800)

    def run_ladder(param, members):
        dirs = {}
        root = work / "convergence" / str(param)
        for setting, inputs in members.items():
            rdir = root / str(setting)
            rdir.mkdir(parents=True, exist_ok=True)
            print(f"  [{param}={setting}] running NWChem in {rdir} ...")
            res = executor.run(inputs, run_cmd, str(rdir))
            print(f"    -> {res.get('status')} rc={res.get('returncode')}")
            dirs[setting] = str(rdir)
        return dirs

    pc = converge_parameters(
        base_inputs={"job.nw": _DECK}, specs=specs,
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
            print(f"  {setting:>10}: {value}")
            if value is None:
                ok = False
        c = s.convergence
        print(f"  -> converged={c.converged} setting={c.setting} "
              f"value={c.value}\n     {c.reason}")

    print("\n" + ("PASS: every rung produced an energy" if ok
                  else "FAIL: some rung produced no energy — NWChem or extraction failed"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
