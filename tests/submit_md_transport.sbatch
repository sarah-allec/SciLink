#!/bin/bash --login
# Live MD-transport run (B1 cadence + B2 replica ensemble + B3 pooling), native:
# SciLink's own executor runs the campaign, and each member runs under MPI on a
# full compute node. Members run SEQUENTIALLY (concurrent fan-out is upstream
# issue #745); each is MPI-parallel, so a ~10 ns replica is ~1-2 h, five ~6-10 h.
#
#   SCILINK_SRC=/path/to/SciLink BENCH_ENV=scilink_ffmd \
#     SCILINK_API_KEY=... SCILINK_MODEL=... \
#     sbatch -p <partition> tests/submit_md_transport.sbatch
#
# Override the engine/launcher at submit time if the site differs:
#   SCILINK_LAMMPS_BIN=/path/to/mpi/lmp SCILINK_MPI_LAUNCHER="srun --mpi=pmix" ...
#
#SBATCH --job-name=md_transport
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64      # <-- set to the compute node's core count
#SBATCH --time=12:00:00
#SBATCH --output=md_transport_%j.out
#SBATCH --error=md_transport_%j.err

set -eo pipefail

module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
# SLURM --export=ALL inherits the login shell's active env, so a plain
# `conda activate` is a no-op while `module load` has already prepended base
# miniconda to PATH — env var right, python base. Drop the inherited env, then
# activate cleanly and hard-prepend the env bin so nothing shadows it.
conda deactivate 2>/dev/null || true
conda activate "${BENCH_ENV:-scilink_ffmd}"   # provides SciLink's python + OpenFF
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "${SCILINK_SRC:?set SCILINK_SRC to the SciLink checkout}"

# The conda `lmp` is a serial MPI-STUBS build; run the external CPU+MPI build
# under srun instead. These are the general overrides default_run_command honors
# (site-specific values stay here, never in the skill).
module load openmpi/5.0.7
export SCILINK_LAMMPS_BIN="${SCILINK_LAMMPS_BIN:-/share/apps/lammps/lammps-29Aug2024/install/bin/lmp_rocky9}"
export SCILINK_MPI_LAUNCHER="${SCILINK_MPI_LAUNCHER:-srun}"
export OMP_NUM_THREADS=1   # pure MPI; no thread oversubscription
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"

echo "node    : $(hostname)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "ntasks  : ${SLURM_NTASKS:-?}  (per-node ${SLURM_NTASKS_PER_NODE:-?})"
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
# LAMMPS engine sanity — the MPI binary must load and launch under the launcher.
echo "lmp bin : ${SCILINK_LAMMPS_BIN}"
echo "launcher: ${SCILINK_MPI_LAUNCHER}"
${SCILINK_MPI_LAUNCHER} -n 1 "${SCILINK_LAMMPS_BIN}" -h >/tmp/lmp_h.$$ 2>&1 \
  && grep -qiE 'MPI v|Large-scale' /tmp/lmp_h.$$ \
  && echo "lmp     : ok ($(grep -m1 -iE 'MPI v' /tmp/lmp_h.$$))" \
  || { echo "lmp     : FAILED to launch under launcher — see below; aborting"; cat /tmp/lmp_h.$$; exit 1; }
rm -f /tmp/lmp_h.$$
echo

python tests/test_md_transport_generate_live.py
