#!/bin/bash
#SBATCH --job-name=uc2_replicas
#SBATCH --account=MATSCI_AI_LDRD
#SBATCH --partition=a100_shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=uc2rep_%A_%a.out
#SBATCH --error=uc2rep_%A_%a.err
#SBATCH --array=0-14

# Independent REPLICAS for the viscosity trend: one array task per (member, seed).
# Each task runs + analyzes one independent copy of a composition into a
# rep_<seed> subdir. Different seeds -> different initial velocities -> the runs
# are independent samples, so the Green-Kubo viscosity trend can be resolved.
#
# Default: 3 members x 5 seeds = 15 tasks (array 0-14).
#   sbatch uc2/submit_uc2_replicas.sbatch
#
# Round 2 (add more seeds after a `converge` check says "run more") — pick NEW,
# non-overlapping seeds and size --array to (n_members * n_seeds - 1):
#   UC2_SEEDS="6 7 8 9" sbatch --array=0-11 uc2/submit_uc2_replicas.sbatch
#
# After a batch finishes, check whether the trend is clear (login node):
#   python uc2/run_uc2.py --stage converge --members 80-20,70-30,60-40 \
#       --runs-dir /rcfs/projects/matsci_ai_ldrd/alle927/uc2/runs

set -euo pipefail

# Members (the composition series) and seeds (independent replicas). Override via
# UC2_MEMBERS / UC2_SEEDS; the array size must equal n_members * n_seeds.
read -ra MEMBERS <<< "${UC2_MEMBERS:-80-20 70-30 60-40}"
read -ra SEEDS   <<< "${UC2_SEEDS:-1 2 3 4 5}"
NMEM=${#MEMBERS[@]}; NSEED=${#SEEDS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= NMEM * NSEED )); then
    echo "array index $IDX out of range for ${NMEM} members x ${NSEED} seeds"; exit 0
fi
MEMBER="${MEMBERS[$(( IDX / NSEED ))]}"
SEED="${SEEDS[$(( IDX % NSEED ))]}"

REPO="${REPO:-/qfs/people/alle927/SciLink/SciLink}"
cd "$REPO"

# ── env: the OpenFF force-field + LAMMPS environment (scilink_ffmd) ──
# Same setup as submit_uc2.sbatch (see its comments for the why).
set +eu
source "${UC2_CONDA_SH:-/share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh}"
if [ -n "${UC2_ENV_SETUP:-}" ] && [ -f "${UC2_ENV_SETUP}" ]; then
    source "${UC2_ENV_SETUP}"
fi
conda activate "${UC2_CONDA_ENV:-/people/alle927/.conda/envs/scilink_ffmd}"
set -eu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${SCILINK_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}:${PYTHONPATH:-}"
_SRC="${PYTHONPATH%%:*}"
echo "scilink : ${_SRC}"
( cd "$_SRC" && echo "pin     : $(git rev-parse --short HEAD 2>/dev/null)  $(git log --oneline -1 2>/dev/null)" ) 2>/dev/null || true

# ── credentials / model (proxy path on Deception) ──
: "${SCILINK_API_KEY:?set SCILINK_API_KEY (proxy key for generation)}"
export SCILINK_BASE_URL="${SCILINK_BASE_URL:?set SCILINK_BASE_URL (the proxy URL)}"
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"

# ── run parameters ──
_LMP="/share/apps/lammps/lammps-29Aug2024/build/lmp_rocky9"
export LAMMPS_CMD="${LAMMPS_CMD:-$_LMP -sf gpu -pk gpu 1 -in {script}}"
export UC2_STAGE_TIMEOUT="${UC2_STAGE_TIMEOUT:-144000}"
export UC2_PRODUCTION_NS="${UC2_PRODUCTION_NS:-25}"

UC2_RUNS_DIR="${UC2_RUNS_DIR:-/rcfs/projects/matsci_ai_ldrd/alle927/uc2/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/rcfs/projects/matsci_ai_ldrd/alle927/uc2/cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
mkdir -p "$UC2_RUNS_DIR" "$MPLCONFIGDIR"

PY="$CONDA_PREFIX/bin/python"

echo "===========  UC2 replica: member $MEMBER  seed $SEED  (-> $UC2_RUNS_DIR)  ==========="
"$PY" uc2/run_uc2.py --stage full    --members "$MEMBER" --seed "$SEED" --runs-dir "$UC2_RUNS_DIR"
"$PY" uc2/run_uc2.py --stage analyze --members "$MEMBER" --seed "$SEED" --runs-dir "$UC2_RUNS_DIR"

echo
echo "Replica ($MEMBER, seed $SEED) done. When the batch finishes, check the trend:"
echo "  $PY uc2/run_uc2.py --stage converge --members ${UC2_MEMBERS:-80-20,70-30,60-40} --runs-dir $UC2_RUNS_DIR"
