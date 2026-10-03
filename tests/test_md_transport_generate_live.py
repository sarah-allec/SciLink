"""Live generate-only check of the MD-transport setup (B1 + B2).

Runs the MD workflow in generate-only mode (no executor, so no hours of LAMMPS)
on a converged-viscosity goal, then inspects what the LLM produced:

  B1 (cadence):   does the production deck log the pressure tensor (pxy/pxz/pyz)
                  at a dense interval?
  B2 (replicas):  did the planner set up an independent-replica ensemble
                  (a parameter sweep → multiple member decks)?

It reports what it finds rather than hard-asserting on LLM choices. B3 (pooled
Green-Kubo) needs executed replicas and is validated separately.

    export SCILINK_MODEL=...   SCILINK_API_KEY=... (or ANTHROPIC_API_KEY)
    python tests/test_md_transport_generate_live.py
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_DIR = REPO_ROOT / "tests" / "_md_transport_generate"

_GOAL = (
    "Compute the shear viscosity of liquid TIP3P water at 298 K and 1 atm via "
    "Green-Kubo. I need a converged value I can stand behind for publication."
)


def _make_orch(base_dir):
    from scilink.agents.sim_agents import (
        SimulationOrchestratorAgent, SimulationMode,
    )
    key = os.environ.get("SCILINK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("set SCILINK_API_KEY or ANTHROPIC_API_KEY")
    kwargs = dict(base_dir=str(base_dir), api_key=key,
                  model_name=os.environ.get("SCILINK_MODEL", "claude-opus-4-6"),
                  simulation_mode=SimulationMode.AUTONOMOUS)
    if os.environ.get("SCILINK_BASE_URL"):
        kwargs["base_url"] = os.environ["SCILINK_BASE_URL"]
    return SimulationOrchestratorAgent(**kwargs)


def _stress_cadence(deck: str):
    """Return a short description of how densely the deck logs the stress."""
    findings = []
    for ln in deck.splitlines():
        s = ln.strip()
        low = s.lower()
        if low.startswith("thermo ") and re.match(r"thermo\s+\d+", low):
            findings.append(("thermo_interval", s))
        if "pxy" in low and ("thermo_style" in low or "fix" in low or "variable" in low):
            findings.append(("stress_line", s))
        if "ave/time" in low and ("pxy" in low or "v_pxy" in low):
            findings.append(("ave_time", s))
    return findings


def main():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    orch = _make_orch(RUN_DIR / "sim")

    # Generate-only: no run_command / executor, so the workflow stops after
    # generation + validation. derive_observables is on for MD (B1).
    raw = orch.tools.functions_map["run_simulation"](
        description=_GOAL, scale="molecular_dynamics", software="lammps",
    )
    result = json.loads(raw)
    out_dir = Path(result.get("output_directory", RUN_DIR))
    print("status:", result.get("status"), "| output:", out_dir)

    decks = sorted(out_dir.rglob("*.lammps")) + sorted(out_dir.rglob("in.*")) \
        + sorted(out_dir.rglob("run.lammps"))
    decks = [d for d in dict.fromkeys(decks) if d.is_file()]
    member_dirs = sorted({p.parent for p in decks})

    print(f"\n{'='*60}\nMD TRANSPORT GENERATE-ONLY FINDINGS\n{'='*60}")
    print(f"decks generated: {len(decks)}")
    print(f"distinct run/member directories: {len(member_dirs)}")
    print(f"  -> B2 replica ensemble: "
          f"{'YES (multiple members)' if len(member_dirs) > 1 else 'no (single run)'}")

    if decks:
        deck = decks[-1].read_text(errors="replace")
        print(f"\nB1 cadence — stress logging in {decks[-1].name}:")
        cad = _stress_cadence(deck)
        if cad:
            for kind, line in cad:
                print(f"  [{kind}] {line}")
        else:
            print("  !! no pxy/pxz/pyz logging found in the deck")

    print("\n(Interpretation: B1 wants pxy/pxz/pyz logged every few steps; "
          "B2 wants several seed-varied members.)")

    # B3: if LAMMPS executed (lmp on PATH, e.g. in an allocation), check the
    # pooled convergence across replica members.
    if result.get("status") == "success":
        print(f"\n{'='*60}\nB3 pooled convergence (requires executed replicas)\n{'='*60}")
        raw2 = orch.tools.functions_map["check_observable_convergence"](
            output_dir=str(out_dir), research_goal=_GOAL)
        conv = json.loads(raw2)
        print("convergence status:", conv.get("status"))
        for prop, ev in (conv.get("properties") or {}).items():
            print(f"  {prop}: value={ev.get('value')} {ev.get('units')} "
                  f"state={ev.get('state')}")
        print("  unconverged:", conv.get("unconverged"),
              "| not_assessed:", conv.get("not_assessed"))
        print("(std_error / n_replicas, when >1 replica ran, are in the "
              "analysis result the GK recipe produced.)")


if __name__ == "__main__":
    main()
