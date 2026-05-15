"""Shared VASP helpers — POSCAR writer (sort=True) and per-cell sbatch
template — lifted from validation/run_panel.py so benchmark/test_dft.py
and benchmark/test_updater.py share one definition.

Conservative defaults that mirror the working bcc_fe runs on Deception:
intel + mkl + openmpi modules, `mpirun vasp_std`, ntasks-per-node=16,
30 min walltime.  Edit ``MODULE_BLOCK`` if your cluster's toolchain differs.
"""
from __future__ import annotations

import os
from typing import Sequence


# Module load block — kept as a string so callers can override per-test
# if needed.  Matches the bcc_fe workflow Sarah confirmed works on
# Deception.
MODULE_BLOCK = """\
module purge
module load intel/2022.1.0
module load mkl/2023.0.0
module load openmpi/5.0.7
"""


def write_poscar(atoms, path: str) -> None:
    """Write POSCAR with sort=True so multi-element cells don't break
    VASP's POTCAR consistency check.

    Bug fix history:  ``ase.build.bulk("MgO", "rocksalt", cubic=True)``
    returns atoms in Mg/O/Mg/O order; ``write(..., sort=False)`` preserved
    that and broke every multi-element validation cell.  See
    ``validation/BUG_NOTES.md``.
    """
    from ase.io import write
    write(path, atoms, format="vasp", direct=True, sort=True)


def write_per_cell_sbatch(rundir: str,
                          job_name: str,
                          elements: Sequence[str],
                          time: str = "00:30:00",
                          ntasks_per_node: int = 16,
                          extra_modules: str = "") -> str:
    """Write the per-cell SLURM script that assembles POTCAR + runs vasp_std.

    Args:
        rundir: directory the script goes into (and SLURM_SUBMIT_DIR at run time).
        job_name: SLURM --job-name.
        elements: unique element symbols in alphabetical or species-line order.
                  POTCAR is built by concatenating ``$VASP_PP_PATH/<el>/POTCAR``
                  for each.
        time: walltime string.
        ntasks_per_node: MPI rank count.
        extra_modules: optional extra ``module load`` lines, e.g. for systems
                       that need a different toolchain.

    Returns:
        Path to the written ``submit.sbatch``.
    """
    elements_str = " ".join(elements)
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={time}
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err

set -e
cd "$SLURM_SUBMIT_DIR"

# ── MPI / MKL toolchain ──────────────────────────────────────────
{MODULE_BLOCK}{extra_modules}
# ── Assemble POTCAR from cluster pseudopotential library ─────────
: "${{VASP_PP_PATH:?set VASP_PP_PATH to the POTCAR library root}}"
rm -f POTCAR
for el in {elements_str}; do
    if [ ! -f "$VASP_PP_PATH/$el/POTCAR" ]; then
        echo "Missing POTCAR for element $el at $VASP_PP_PATH/$el/POTCAR" >&2
        exit 1
    fi
    cat "$VASP_PP_PATH/$el/POTCAR" >> POTCAR
done

mpirun vasp_std
"""
    path = os.path.join(rundir, "submit.sbatch")
    with open(path, "w") as f:
        f.write(script)
    return path


def read_relaxed_atoms(rundir: str):
    """Return ASE Atoms from CONTCAR; raises FileNotFoundError if missing
    or zero-byte (the symptom we get when VASP refused at startup, e.g.
    MgO POSCAR/POTCAR mismatch)."""
    from ase.io import read
    contcar = os.path.join(rundir, "CONTCAR")
    if not os.path.exists(contcar) or os.path.getsize(contcar) == 0:
        raise FileNotFoundError(
            f"CONTCAR missing or empty in {rundir} — VASP did not produce "
            f"a relaxed structure")
    return read(contcar, format="vasp")


def lattice_a(atoms) -> float:
    """Cubic a parameter (cellpar[0]).  For non-cubic cells callers should
    compute the appropriate observable themselves."""
    return float(atoms.cell.cellpar()[0])


def final_energy(rundir: str) -> float | None:
    """Total energy in eV from OSZICAR last line, or None if unparseable."""
    osz = os.path.join(rundir, "OSZICAR")
    if not os.path.exists(osz):
        return None
    last_line = None
    with open(osz) as f:
        for line in f:
            if " E0=" in line or " F=" in line:
                last_line = line
    if not last_line:
        return None
    # OSZICAR final line: "  10 F= -.XXX E0= -.XXX d E =-..."
    try:
        for tok in last_line.split():
            if tok.startswith("E0="):
                return float(tok.split("=", 1)[1])
    except Exception:
        pass
    return None
