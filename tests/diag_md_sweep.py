"""Diagnostic: does the MD planner set up a seed ensemble, and does the campaign
expand to multiple members? Generate-only (executor=None) — NO LAMMPS, a few
minutes — so it isolates planning + campaign expansion from execution.

Prints (via the new logs in md_simulation_agent):
  "Planner requested a sweep: variable_parameter=... variable_values=[...] ..."
  "Campaign expanded to N member(s) over 'velocity seed' = [...]"

    export SCILINK_MODEL=...  SCILINK_API_KEY=... (or ANTHROPIC_API_KEY)
    python tests/diag_md_sweep.py
"""

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_md_transport_generate_live import _make_orch, _GOAL  # noqa: E402

WORK = REPO_ROOT / "tests" / "_md_sweep_diag"


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    orch = _make_orch(WORK / "cfg")   # resolves the proxy key/base_url/model

    from scilink.agents.sim_agents.simulation_pipeline import run_complete_workflow

    # executor=None → generation + validation only, no LAMMPS. derive_observables
    # on (MD) so the planner gets the transport cadence/ensemble contract.
    res = run_complete_workflow(
        _GOAL,
        scale="molecular_dynamics", software="lammps", structure_class="condensed",
        output_dir=str(WORK / "run"),
        api_key=orch.api_key, base_url=orch.base_url, model_name=orch.model_name,
        mp_api_key=orch.mp_api_key,
        derive_observables=True, validate=False,
    )
    print("\nfinal_status:", res.get("final_status"))
    if res.get("final_status") == "failed_force_field":
        print("force_field error:", (res.get("force_field") or {}).get("message"))
        print("(FF parameterization failed before planning — likely the wrong "
              "env: OpenFF must be importable. Run in scilink_ffmd.)")
    gen = res.get("input_generation") or {}
    print("is_campaign:", gen.get("is_campaign"), "| stages:", len(gen.get("stages") or []))
    # Count member decks produced on disk.
    members = sorted((WORK / "run").rglob("production/*/"))
    print("member directories produced:", len(members))
    for m in members[:10]:
        print("   ", m.name)


if __name__ == "__main__":
    main()
