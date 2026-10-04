#!/bin/bash --login
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

echo "node    : $(hostname)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "lmp     : $(command -v lmp || echo MISSING)"
echo "key     : $([ -n "${SCILINK_API_KEY:-${ANTHROPIC_API_KEY:-}}" ] && echo set || echo MISSING)"
# Env sanity — the batch job must land in the env where OpenFF imports, or
# every member fails force-field parameterization. Fail loud, up front.
echo "condaenv: ${CONDA_DEFAULT_ENV:-none}"
echo "python  : $(command -v python)"
python -c "import openff.toolkit; print('openff  : ok')" \
  || { echo "openff  : MISSING (wrong env — FF will fail; aborting)"; exit 1; }
# Known drift: pydantic >=2.12 breaks openff-interchange serialization even
# when OpenFF imports (failed_force_field, zero-byte interchange). Want <2.12.
python -c "import pydantic,sys; v=tuple(int(x) for x in pydantic.VERSION.split('.')[:2]); print('pydantic:', pydantic.VERSION); sys.exit(0 if v < (2,12) else 1)" \
  || { echo "pydantic: >=2.12 — openff-interchange will fail; run 'pip install \"pydantic>=2.11,<2.12\"'; aborting"; exit 1; }
echo

python tests/test_md_transport_generate_live.py
