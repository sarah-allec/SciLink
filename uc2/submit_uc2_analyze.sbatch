#!/bin/bash
#SBATCH --job-name=uc2_analyze
#SBATCH --account=MATSCI_AI_LDRD
#SBATCH --partition=slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=uc2an_%A_%a.out
#SBATCH --error=uc2an_%A_%a.err
#SBATCH --array=0-5

# Re-analyze FINISHED replicas in parallel — one array task per replica, CPU only
# (no MD, no GPU). Recomputes density/viscosity/T1 from each run's EXISTING output
# using the current analysis skills (e.g. the fit-based Green-Kubo estimator), and
# overwrites that replica's results.json.
#
# Size the array to the number of finished replicas (those with a results.json):
#   RUNS=/rcfs/projects/matsci_ai_ldrd/alle927/uc2/runs
#   ls -d $RUNS/{80-20,70-30,60-40}/rep_*/results.json | wc -l      # -> N
#   sbatch --array=0-$((N-1)) uc2/submit_uc2_analyze.sbatch
#
# The analysis uses the LLM proxy for codegen, so the CPU nodes must reach it.
# If the 'slurm' partition can't, override:  sbatch -p <cpu-part-with-proxy> ...

set -euo pipefail

REPO="${REPO:-/qfs/people/alle927/SciLink/SciLink}"
cd "$REPO"
UC2_RUNS_DIR="${UC2_RUNS_DIR:-/rcfs/projects/matsci_ai_ldrd/alle927/uc2/runs}"

# Stable, sorted list of FINISHED replicas (a results.json means the pipeline
# completed for that seed); still-running replicas have none and are skipped.
read -ra MEMBERS <<< "${UC2_MEMBERS:-80-20 70-30 60-40}"
REPS=()
for m in "${MEMBERS[@]}"; do
  for r in "$UC2_RUNS_DIR/$m"/rep_*; do
    [ -f "$r/results.json" ] && REPS+=("$r")
  done
done
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#REPS[@]} )); then
  echo "index $IDX >= ${#REPS[@]} finished replicas; nothing to do"; exit 0
fi
REP="${REPS[$IDX]}"
MEMBER="$(basename "$(dirname "$REP")")"
SEED="${REP##*rep_}"

# ---- env: same OpenFF/LAMMPS env as the run sbatch (see submit_uc2.sbatch) ----
set +eu
source "${UC2_CONDA_SH:-/share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh}"
if [ -n "${UC2_ENV_SETUP:-}" ] && [ -f "${UC2_ENV_SETUP}" ]; then source "${UC2_ENV_SETUP}"; fi
conda activate "${UC2_CONDA_ENV:-/people/alle927/.conda/envs/scilink_ffmd}"
set -eu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SCILINK_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}:${PYTHONPATH:-}"

: "${SCILINK_API_KEY:?set SCILINK_API_KEY (proxy key for analysis codegen)}"
export SCILINK_BASE_URL="${SCILINK_BASE_URL:?set SCILINK_BASE_URL (the proxy URL)}"
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/rcfs/projects/matsci_ai_ldrd/alle927/uc2/cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"; mkdir -p "$MPLCONFIGDIR"

PY="$CONDA_PREFIX/bin/python"
echo "=== re-analyze  member $MEMBER  seed $SEED  ($REP) ==="
"$PY" uc2/run_uc2.py --stage analyze --members "$MEMBER" --seed "$SEED" \
    --runs-dir "$UC2_RUNS_DIR"
