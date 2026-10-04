#!/bin/bash --login
# Lightweight diagnostic: MD seed-ensemble planning + campaign expansion,
# generate-only (NO LAMMPS). A few minutes — fire and forget.
#
#   SCILINK_SRC=/path/to/SciLink BENCH_ENV=scilink_ffmd \
#     SCILINK_API_KEY=... SCILINK_MODEL=... \
#     sbatch -p <partition> tests/submit_md_sweep_diag.sbatch
#
#SBATCH --job-name=md_sweep_diag
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=md_sweep_diag_%j.out
#SBATCH --error=md_sweep_diag_%j.err

set -eo pipefail

module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate "${BENCH_ENV:-scilink_ffmd}"
cd "${SCILINK_SRC:?set SCILINK_SRC to the SciLink checkout}"

export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"
echo "branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "key     : $([ -n "${SCILINK_API_KEY:-${ANTHROPIC_API_KEY:-}}" ] && echo set || echo MISSING)"
# Env sanity — the batch job must be in the env where OpenFF imports.
echo "condaenv: ${CONDA_DEFAULT_ENV:-none}"
echo "python  : $(command -v python)"
python -c "import openff.toolkit; print('openff  : ok')" || echo "openff  : MISSING (wrong env — FF will fail)"
echo

python tests/diag_md_sweep.py
