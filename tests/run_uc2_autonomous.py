"""Let SciLink loose on use case 2 — the META orchestrator, autonomous.

Hands the meta orchestrator ONE free-text electrolyte goal and lets it run:
it routes to the simulation specialist, which builds + equilibrates the box,
runs MD, analyzes, and drives the step-8 convergence loop (escalate sampling
on an under-sampled transport observable, re-run, re-check; stop + diagnose a
method limit). No force field, water model, protocol, or sampling depth is
specified — those are SciLink's to choose.

    export SCILINK_API_KEY=...  SCILINK_MODEL=claude-opus-4-8-project
    export SCILINK_LAMMPS_BIN=...  SCILINK_MPI_LAUNCHER="mpirun -np <N>"
    export SCILINK_RUN_TIMEOUT=21600          # per-run wall floor (s)
    python tests/run_uc2_autonomous.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# One composition to start (80:20). Free-text: name the system, the state
# point, and the properties wanted — then STOP. How to reach a trustworthy
# value (sampling strategy, when to escalate, when the limit is the method) is
# SciLink's own call, driven by its convergence-loop guidance; spelling it out
# here would make the run follow the prompt instead of acting autonomously.
GOAL = (
    "Build and equilibrate a 1 M zinc triflate (Zn(OTf)2) electrolyte in a "
    "water / ethyl-isopropyl-sulfone solvent at an 80:20 water-to-sulfone "
    "volume ratio. Run molecular dynamics and compute the mass density, the "
    "shear viscosity by the Green-Kubo method, the water self-diffusion "
    "coefficient from the mean-squared displacement, and the water 1H "
    "spin-lattice (T1) relaxation time. I need a shear viscosity I can stand "
    "behind for publication."
)


def main():
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode,
    )
    key = os.environ.get("SCILINK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("set SCILINK_API_KEY or ANTHROPIC_API_KEY")

    kwargs = dict(
        base_dir=str(REPO_ROOT / "tests" / "_uc2_autonomous"),
        api_key=key,
        model_name=os.environ.get("SCILINK_MODEL", "claude-opus-4-8-project"),
        meta_mode=MetaMode.AUTONOMOUS,
    )
    if os.environ.get("SCILINK_BASE_URL"):
        kwargs["base_url"] = os.environ["SCILINK_BASE_URL"]
    if os.environ.get("SCILINK_META_MAX_ITERS"):
        kwargs["max_iterations"] = int(os.environ["SCILINK_META_MAX_ITERS"])

    meta = MetaOrchestratorAgent(**kwargs)
    print("=== GOAL ===\n" + GOAL, flush=True)
    print("\n=== LAUNCHING META ORCHESTRATOR (autonomous) ===\n", flush=True)
    out = meta.chat(GOAL)
    print("\n=== META FINAL RESPONSE ===\n" + str(out))


if __name__ == "__main__":
    main()
