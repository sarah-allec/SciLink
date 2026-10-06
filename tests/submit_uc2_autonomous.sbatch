#!/bin/bash --login
# Use case 2, autonomous: the META orchestrator, given one free-text Zn(OTf)2
# electrolyte goal, runs the whole thing itself (build -> MD -> analyze ->
# convergence loop). SciLink's own executor runs LAMMPS under MPI on a full
# node. Long wall + a per-run timeout floor because the convergence loop may
# escalate to long production runs; the loop also has its own sampling budget.
#
#   SCILINK_SRC=/path/to/SciLink BENCH_ENV=scilink_ffmd \
#     SCILINK_API_KEY=... SCILINK_MODEL=... \
#     sbatch -p <partition> --ntasks-per-node=<cores> tests/submit_uc2_autonomous.sbatch
#
#SBATCH --job-name=uc2_auto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64      # <-- set to the compute node's core count
#SBATCH --time=7-00:00:00         # full week — the autonomous loop may run long
#SBATCH --output=uc2_auto_%j.out
#SBATCH --error=uc2_auto_%j.err

set -eo pipefail

# Modules FIRST, conda LAST (loading modules after `conda activate` de-inits the
# env). openmpi is compiler-dependent, so gcc loads before it. mpirun, not srun
# (this site's srun has no pmix plugin). The conda `lmp` is MPI-STUBS, so point
# at the external CPU+MPI build.
module load gcc/11.5.0 openmpi/5.0.7
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate "${BENCH_ENV:-scilink_ffmd}"
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "${SCILINK_SRC:?set SCILINK_SRC to the SciLink checkout}"

export SCILINK_LAMMPS_BIN="${SCILINK_LAMMPS_BIN:-/share/apps/lammps/lammps-29Aug2024/install/bin/lmp_rocky9}"
export SCILINK_MPI_LAUNCHER="${SCILINK_MPI_LAUNCHER:-mpirun -np ${SLURM_NTASKS}}"
export OMP_NUM_THREADS=1
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"
# Per-run wall floor so a long production run isn't killed at the 1 h default
# (the autonomous loop calls run_simulation itself and can't pass the arg).
export SCILINK_RUN_TIMEOUT="${SCILINK_RUN_TIMEOUT:-86400}"   # 24 h/run — a
# single staged run (equil + long GK production, one lmp invocation) can take
# most of a day for a big box, and the loop may escalate to longer ones; the
# 7-day wall is the real bound, so keep the per-run floor from truncating.

echo "node    : $(hostname)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git log --oneline -1)"
echo "ntasks  : ${SLURM_NTASKS:-?}  (per-node ${SLURM_NTASKS_PER_NODE:-?})"
echo "key     : $([ -n "${SCILINK_API_KEY:-${ANTHROPIC_API_KEY:-}}" ] && echo set || echo MISSING)"
echo "condaenv: ${CONDA_DEFAULT_ENV:-none}"
echo "python  : $(command -v python)"
echo "mpirun  : $(command -v mpirun || echo MISSING)"
echo "run_to  : ${SCILINK_RUN_TIMEOUT}s per run"
python -c "import openff.toolkit; print('openff  : ok')" \
  || { echo "openff  : MISSING (wrong env — FF will fail; aborting)"; exit 1; }
python -c "import pydantic,sys; v=tuple(int(x) for x in pydantic.VERSION.split('.')[:2]); print('pydantic:', pydantic.VERSION); sys.exit(0 if v < (2,12) else 1)" \
  || { echo "pydantic: >=2.12 — openff-interchange will fail; aborting"; exit 1; }
# Engine sanity: confirm real parallel ranks before the (long) campaign.
echo "lmp bin : ${SCILINK_LAMMPS_BIN}"
echo "launcher: ${SCILINK_MPI_LAUNCHER}"
_probe="mpi_probe_$$.in"
printf 'units lj\natom_style atomic\nlattice fcc 0.8\nregion b block 0 8 0 8 0 8\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nrun 0\n' > "$_probe"
_out=$(${SCILINK_MPI_LAUNCHER} "${SCILINK_LAMMPS_BIN}" -in "$_probe" 2>&1)
_nproc=$(printf '%s\n' "$_out" | grep -oE 'on [0-9]+ procs' | grep -oE '[0-9]+' | head -1)
rm -f "$_probe"
echo "lmp     : launched on ${_nproc:-?} procs"
[ "${_nproc:-1}" -gt 1 ] \
  || { echo "lmp     : NOT parallel (want >1 procs) — aborting"; printf '%s\n' "$_out" | tail -20; exit 1; }
echo

# -u: unbuffered stdout so the meta's print() handoffs appear live in %j.out
# (logging already streams to %j.err).
python -u tests/run_uc2_autonomous.py
