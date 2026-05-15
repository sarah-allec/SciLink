#!/bin/bash
#SBATCH --job-name=bench_test_mlip
#SBATCH --partition=a100_shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=bench_test_mlip_%j.out
#SBATCH --error=bench_test_mlip_%j.err

# Cluster driver for benchmark/test_mlip.py.
# Runs `generate` (LLM + deploy_pretrained — needs GPU for mace-torch /
# chgnet model load) and then sbatches every cell's per-cell MD job.
#
# What runs WHERE:
#   * this GPU job: `generate` (LLM + initial model deploy + script gen)
#   * MLIP-MD runs: submitted as separate GPU jobs (one per cell)
#   * `collect`: run by hand once those MD jobs finish

set -euo pipefail

REPO="${REPO:-/qfs/people/alle927/SciLink/SciLink}"
cd "$REPO"

# ── env setup ────────────────────────────────────────────────────
source /people/alle927/scilink_mace_env.sh

# Fail fast on config:
: "${SCILINK_API_KEY:?set SCILINK_API_KEY  (generate needs LLM access)}"

PY="$CONDA_PREFIX/bin/python"

echo "===================  test_mlip :: generate  ==================="
if ! "$PY" -m benchmark.runner test_mlip --phase generate; then
    echo "!! generate phase failed — aborting before sbatch sweep" >&2
    exit 1
fi

echo
echo "===================  test_mlip :: submit    ==================="
"$PY" -m benchmark.runner test_mlip --phase submit

echo
echo "All per-cell MLIP-MD jobs submitted (see"
echo "  benchmark/outputs/test_mlip/manifest.json"
echo "for job IDs and the backend each cell got).  Once they finish:"
echo
echo "  $PY -m benchmark.runner test_mlip --phase collect"
