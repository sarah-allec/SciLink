#!/bin/bash
# Live DFT parameter-convergence test (real VASP, in-allocation).
#
#   SCILINK_SRC=/path/to/SciLink BENCH_ENV=scilink_dev SYSTEM=cu \
#     sbatch -p <partition> tests/submit_dft_convergence.sbatch
#
# Runs the ENCUT + k-point convergence sweeps on a small cell with real vasp_std
# and prints the plateau report. VASP runs in-process (LocalExecutor) within this
# allocation, mirroring the MD live test.
#
#SBATCH --job-name=dft_conv
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=02:00:00
#SBATCH --output=dft_conv_%j.out
#SBATCH --error=dft_conv_%j.err

set -eo pipefail

# VASP toolchain (matches the benchmark's run_vasp.sh).
module load intel/2022.1.0
module load mkl/2023.0.0
module load openmpi/5.0.7

# conda is not on PATH in a batch shell; load the miniconda module and source
# its init the way it's started interactively on this cluster.
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate "${BENCH_ENV:-scilink_dev}"
cd "${SCILINK_SRC:?set SCILINK_SRC to the SciLink checkout}"

# Per-element POTCAR root (dir containing Cu/POTCAR, Si/POTCAR, ...).
export VASP_PP_PATH="${VASP_PP_PATH:-/people/alle927/VASP_POT/potpaw_PBE}"
export VASP_RUN_CMD="${VASP_RUN_CMD:-mpirun vasp_std}"

echo "node   : $(hostname)"
echo "branch : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "vasp   : $(command -v vasp_std || echo MISSING)"
echo "PP     : $VASP_PP_PATH"
echo

python tests/test_dft_convergence_live.py --system "${SYSTEM:-cu}"
