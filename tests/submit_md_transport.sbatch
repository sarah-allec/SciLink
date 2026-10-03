#!/bin/bash
# Live MD-transport run (B1 cadence + B2 replica ensemble + B3 pooling), in an
# allocation so the real LAMMPS replicas run on a compute node, not a login node.
#
#   SCILINK_SRC=/path/to/SciLink BENCH_ENV=scilink_ffmd \
#     SCILINK_API_KEY=... SCILINK_MODEL=... \
#     sbatch -p <partition> tests/submit_md_transport.sbatch
#
#SBATCH --job-name=md_transport
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=06:00:00
#SBATCH --output=md_transport_%j.out
#SBATCH --error=md_transport_%j.err

set -eo pipefail

module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate "${BENCH_ENV:-scilink_ffmd}"   # has lmp + OpenFF; pydantic pinned
cd "${SCILINK_SRC:?set SCILINK_SRC to the SciLink checkout}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"

echo "node   : $(hostname)"
echo "branch : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "lmp    : $(command -v lmp || echo MISSING)"
echo "key    : $([ -n "${SCILINK_API_KEY:-${ANTHROPIC_API_KEY:-}}" ] && echo set || echo MISSING)"
echo

python tests/test_md_transport_generate_live.py
