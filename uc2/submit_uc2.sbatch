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
# A batch shell runs neither interactive conda-init nor the `start_conda` alias,
# so `conda` is absent unless we source the profile script ourselves. This is
# what start_conda does. Activate the env by FULL PATH — it lives under
# ~/.conda/envs, not the miniconda's own envs, so activate-by-name may miss it.
# UC2_ENV_SETUP stays an OPTIONAL hook, sourced only if it exists.
# LD_LIBRARY_PATH puts the conda libs first: fixes the libstdc++/CXXABI mismatch
# that otherwise makes NAGL (force-field charges) and matplotlib fail to import.
# conda's own activate.d scripts (binutils, etc.) reference unbound variables
# and can return non-zero, so relax nounset/errexit just around activation and
# restore the strict flags afterward.
set +eu
source "${UC2_CONDA_SH:-/share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh}"
if [ -n "${UC2_ENV_SETUP:-}" ] && [ -f "${UC2_ENV_SETUP}" ]; then
    source "${UC2_ENV_SETUP}"
fi
conda activate "${UC2_CONDA_ENV:-/people/alle927/.conda/envs/scilink_ffmd}"
set -eu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ── credentials / model (proxy path on Deception) ──
: "${SCILINK_API_KEY:?set SCILINK_API_KEY (proxy key for generation)}"
export SCILINK_BASE_URL="${SCILINK_BASE_URL:?set SCILINK_BASE_URL (the proxy URL)}"
export SCILINK_MODEL="${SCILINK_MODEL:-claude-opus-4-8-project}"

# ── run parameters ──
# Cluster-standard LAMMPS (29 Aug 2024), GPU-accelerated, verified to carry the
# packages this electrolyte deck needs: MOLECULE (bond/angle/dihedral),
# KSPACE (pppm), RIGID (fix shake), + GPU package. Override LAMMPS_CMD to change.
_LMP="/share/apps/lammps/lammps-29Aug2024/build/lmp_rocky9"
export LAMMPS_CMD="${LAMMPS_CMD:-$_LMP -sf gpu -pk gpu 1 -in {script}}"
export UC2_STAGE_TIMEOUT="${UC2_STAGE_TIMEOUT:-72000}"   # 20 h per stage cap
export UC2_PRODUCTION_NS="${UC2_PRODUCTION_NS:-25}"      # human-set viscosity length

# ── output + caches on the PROJECT filesystem, never the quota'd /qfs home ──
# /qfs/people fills up and mkdir/caches fail there; rcfs has the room. The MLIP-
# free FF/MD stack still writes matplotlib/torch caches to $HOME/.cache unless
# redirected, which also fails on a full home.
UC2_RUNS_DIR="${UC2_RUNS_DIR:-/rcfs/projects/matsci_ai_ldrd/alle927/uc2/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/rcfs/projects/matsci_ai_ldrd/alle927/uc2/cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
mkdir -p "$UC2_RUNS_DIR" "$MPLCONFIGDIR"

PY="$CONDA_PREFIX/bin/python"

echo "=================  UC2 member: $MEMBER  (runs -> $UC2_RUNS_DIR)  ================="
"$PY" uc2/run_uc2.py --stage full    --members "$MEMBER" --runs-dir "$UC2_RUNS_DIR"
"$PY" uc2/run_uc2.py --stage analyze --members "$MEMBER" --runs-dir "$UC2_RUNS_DIR"

echo
echo "Member $MEMBER done. Once all four finish, validate on the login node:"
echo "  $PY uc2/run_uc2.py --stage validate --runs-dir $UC2_RUNS_DIR"
