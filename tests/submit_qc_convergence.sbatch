#!/bin/bash
# Live QC basis-convergence test (real NWChem, in-allocation).
#
#   SCILINK_SRC=/path/to/SciLink BENCH_ENV=scilink_dev \
#     sbatch -p <partition> tests/submit_qc_convergence.sbatch
#
# Runs the basis-set convergence sweep on a small water molecule with real
# NWChem and prints the plateau report. NWChem runs in-process (LocalExecutor)
# within this allocation.
#
#SBATCH --job-name=qc_conv
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --output=qc_conv_%j.out
#SBATCH --error=qc_conv_%j.err

set -eo pipefail

# Python/driver env only here — NWChem's own toolchain (gcc, openmpi, nwchem +
# NWCHEM_BASIS_LIBRARY) is set up per run inside tests/run_nwchem.sh, so it does
# not collide with the conda env.
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate "${BENCH_ENV:-scilink_dev}"
cd "${SCILINK_SRC:?set SCILINK_SRC to the SciLink checkout}"

# Each ladder rung runs NWChem through the wrapper (loads modules + basis lib).
export NWCHEM_RUN_CMD="${NWCHEM_RUN_CMD:-bash --login ${SCILINK_SRC}/tests/run_nwchem.sh}"

echo "node   : $(hostname)"
echo "branch : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "nwchem : $(command -v nwchem || echo MISSING)"
echo

python tests/test_qc_convergence_live.py
