"""run_simulation turns on observable derivation for MD, not for static scales.

This is Workstream B's cadence enabler (#682): deriving the observable
contract is what carries a transport goal's dense sampling-cadence requirement
into planning and deck generation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402


def _make_tools():
    orch = MagicMock()
    orch.api_key = "k"
    orch.base_url = None
    orch.model_name = "claude-opus-4-8"
    orch.base_dir = "/tmp/t"
    orch.mp_api_key = None
    orch.futurehouse_api_key = None
    orch.hpc_connection = None
    orch.generated_structures = []
    orch.routing_decision = {}
    orch._structure_counter = 1
    orch.active_skill_and_domain = MagicMock(return_value=(None, None))
    from scilink.agents.sim_agents.simulation_orchestrator_tools import (
        SimulationOrchestratorTools,
    )
    return SimulationOrchestratorTools(orch)


def _capture_derive(monkeypatch, tmp_path, scale, software):
    captured = {}

    def fake_wf(description, **kwargs):
        captured.update(kwargs)
        return {"final_status": "success", "structure_generation": {}}

    monkeypatch.setattr(sp, "run_complete_workflow", fake_wf)
    structure = tmp_path / "structure.extxyz"
    structure.write_text("dummy\n")
    tools = _make_tools()
    tools.functions_map["run_simulation"](
        description="compute a property",
        scale=scale, software=software, structure_file=str(structure),
    )
    return captured


def test_md_enables_derive_observables(tmp_path, monkeypatch):
    captured = _capture_derive(monkeypatch, tmp_path,
                               scale="molecular_dynamics", software="lammps")
    assert captured.get("derive_observables") is True


def test_static_scale_does_not_derive(tmp_path, monkeypatch):
    captured = _capture_derive(monkeypatch, tmp_path,
                               scale="periodic_dft", software="vasp")
    assert captured.get("derive_observables") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
