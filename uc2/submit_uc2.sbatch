#!/bin/bash
#SBATCH --job-name=uc2_znotf
#SBATCH --account=MATSCI_AI_LDRD
#SBATCH --partition=a100_shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=uc2_%A_%a.out
#SBATCH --error=uc2_%A_%a.err
#SBATCH --array=0-3

# One array task per composition. Each generates + runs + refines a single
# member end to end (autonomous), then analyzes its own output.
#
#   sbatch uc2/submit_uc2.sbatch                # all four (array 0-3)
#   sbatch --array=0 uc2/submit_uc2.sbatch      # just 80-20
#
# Review the decks first, interactively, on the login node:
#   python uc2/run_uc2.py --stage preview --members 80-20
# and only launch this once the box / FF / observable contract look right.

set -euo pipefail

MEMBERS=(80-20 70-30 60-40 50-50)
MEMBER="${MEMBERS[${SLURM_ARRAY_TASK_ID:-0}]}"

REPO="${REPO:-/qfs/people/alle927/SciLink/SciLink}"
cd "$REPO"

# ── env: the OpenFF force-field + LAMMPS environment (scilink_ffmd) ──
# Point UC2_ENV_SETUP at your env-activation script. It must leave a working
# `python` (scilink + OpenFF Interchange) and `lmp` on PATH. The scilink_ffmd
# env also needs its conda libs on the loader path and numpy<2.3 (see the
# reference notes); fold those into the setup script.
source "${UC2_ENV_SETUP:-/people/alle927/scilink_ffmd_env.sh}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ── credentials / model (proxy path on Deception) ──
: "${SCILINK_API_KEY:?set SCILINK_API_KEY (proxy key for generation)}"
export SCILINK_BASE_URL="${SCILINK_BASE_URL:?set SCILINK_BASE_URL (the proxy URL)}"
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"

# ── run parameters ──
export LAMMPS_CMD="${LAMMPS_CMD:-lmp -in {script}}"
export UC2_STAGE_TIMEOUT="${UC2_STAGE_TIMEOUT:-72000}"   # 20 h per stage cap

PY="$CONDA_PREFIX/bin/python"

echo "=================  UC2 member: $MEMBER  ================="
"$PY" uc2/run_uc2.py --stage full    --members "$MEMBER"
"$PY" uc2/run_uc2.py --stage analyze --members "$MEMBER"

echo
echo "Member $MEMBER done. Once all four finish, validate on the login node:"
echo "  $PY uc2/run_uc2.py --stage validate"
