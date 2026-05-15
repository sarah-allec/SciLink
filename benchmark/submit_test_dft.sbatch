#!/bin/bash
#SBATCH --job-name=bench_test_dft
#SBATCH --partition=a100_shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=bench_test_dft_%j.out
#SBATCH --error=bench_test_dft_%j.err

# Cluster driver for benchmark/test_dft.py.
# Runs `generate` (LLM-driven INCAR/KPOINTS generation; GPU helps if
# the LLM proxy needs it) then sbatches every cell's per-cell job.
#
# What runs WHERE:
#   * this GPU job: only the `generate` phase (LLM calls)
#   * VASP relaxations: submitted as their own CPU jobs (one per cell)
#   * `collect`: run by hand once the VASP jobs finish (see tail)

set -euo pipefail

REPO="${REPO:-/qfs/people/alle927/SciLink/SciLink}"
cd "$REPO"

# ── env setup (same flow as the lattice-constant panel) ──
source /people/alle927/scilink_mace_env.sh

# Fail fast on config rather than after 8 VASP jobs queue:
: "${SCILINK_API_KEY:?set SCILINK_API_KEY  (generate phase needs LLM access)}"
: "${VASP_PP_PATH:?set VASP_PP_PATH       (per-cell jobs assemble POTCAR from it)}"

PY="$CONDA_PREFIX/bin/python"

echo "===================  test_dft :: generate  ==================="
if ! "$PY" -m benchmark.runner test_dft --phase generate; then
    echo "!! generate phase failed — aborting before sbatch sweep" >&2
    exit 1
fi

echo
echo "===================  test_dft :: submit   ==================="
"$PY" -m benchmark.runner test_dft --phase submit

echo
echo "All per-cell VASP jobs submitted (see manifest at"
echo "  benchmark/outputs/test_dft/manifest.json"
echo "for job IDs).  Once they finish, run:"
echo
echo "  $PY -m benchmark.runner test_dft --phase collect"
