"""The pipeline's post-run advisory helpers surface a human-approval advisory
(reparameterize, or escalate the potential) — never auto-run.

The advisor's LLM is replaced with a fake, so no API key / model / simulation.
Path A (_post_run_advisory) adapts a run-level failure_class verdict; Path B
(_post_run_reference_validation) validates caller-supplied observables against
their references with the deterministic numeric judge.
"""

import scilink.agents.sim_agents.critics as critics_mod
from scilink.agents.sim_agents.simulation_pipeline import (
    _post_run_advisory, _post_run_reference_validation)


class _FakeAdvisor:
    """Stands in for ReparameterizationAdvisor — always escalates to an MLIP."""
    def __init__(self, *a, **k):
        pass

    def advise(self, flagged, system_description="", backend=""):
        return {"recommended_action": "escalate_potential", "suggested_method": "mlip",
                "diagnosis": "d", "detail": "x", "rationale": "r",
                "requires_human": True}


class _RaisingAdvisor:
    def __init__(self, *a, **k):
        raise RuntimeError("no model available")


def test_post_run_advisory_shapes_escalation(monkeypatch):
    monkeypatch.setattr(critics_mod, "ReparameterizationAdvisor", _FakeAdvisor)
    adv = _post_run_advisory("force_field", "a transport property is off ~5x",
                             "T1", "some system",
                             api_key=None, base_url=None, model_name="m")
    assert adv["status"] == "advise_method_escalation"
    assert adv["auto_run"] is False and adv["requires_human_approval"] is True
    assert adv["suggested_next_step"]["method"] == "deploy_pretrained"


def test_post_run_advisory_degrades_to_none(monkeypatch):
    # No model available → best-effort advisory returns None (pipeline keeps its
    # existing warning-only behaviour), never raises.
    monkeypatch.setattr(critics_mod, "ReparameterizationAdvisor", _RaisingAdvisor)
    assert _post_run_advisory("force_field", "why", "T1", "sys",
                              api_key=None, base_url=None, model_name="m") is None


def test_reference_validation_drives_advisory_on_failure(monkeypatch):
    monkeypatch.setattr(critics_mod, "ReparameterizationAdvisor", _FakeAdvisor)
    obs = [{"observable": "density", "value": 1.24, "reference": 1.22},   # ok
           {"observable": "viscosity", "value": 0.46, "reference": 2.70}]  # 5x low
    out = _post_run_reference_validation(
        obs, "some system", api_key=None, base_url=None, model_name="m")
    assert out["failed"] == ["viscosity"] and out["passed"] == ["density"]
    assert out["advisory"]["status"] == "advise_method_escalation"
    assert out["advisory"]["auto_run"] is False


def test_reference_validation_no_advisory_when_all_pass(monkeypatch):
    monkeypatch.setattr(critics_mod, "ReparameterizationAdvisor", _FakeAdvisor)
    obs = [{"observable": "density", "value": 1.23, "reference": 1.22}]
    out = _post_run_reference_validation(
        obs, "sys", api_key=None, base_url=None, model_name="m")
    assert out["failed"] == [] and "advisory" not in out
    assert out["prediction_warranted"] is True
