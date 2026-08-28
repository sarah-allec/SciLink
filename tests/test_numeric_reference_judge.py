"""numeric_reference_judge: a deterministic judge_fn for run_validation_panel.

Compares each observable's computed value to its reference within a relative
tolerance. No LLM. General across observables.
"""

from scilink.agents.sim_agents.reference_validation import (
    numeric_reference_judge, run_validation_panel)


def _per(observations, **kw):
    return {p["observable"]: p for p in
            numeric_reference_judge(observations, **kw)["per_observable"]}


def test_within_tolerance_is_consistent():
    p = _per([{"observable": "density", "value": 1.24, "reference": 1.22}])
    assert p["density"]["consistent"] is True


def test_large_deviation_flags_inconsistent():
    # viscosity 5x too low, well past the default 15% tolerance
    p = _per([{"observable": "viscosity", "value": 0.46, "reference": 2.70}])
    assert p["viscosity"]["consistent"] is False
    assert p["viscosity"]["rel_error"] > 0.8


def test_direction_records_under_prediction():
    # A computed value below its reference must read as UNDER-prediction, so a
    # downstream diagnosis cannot invert the physics.
    p = _per([{"observable": "viscosity", "value": 0.46, "reference": 2.70,
               "units": "mPa*s"}])["viscosity"]
    assert p["direction"] == "under"
    assert "under-predicts" in p["reasoning"]
    assert "0.46" in p["reasoning"] and "2.7" in p["reasoning"]


def test_direction_records_over_prediction():
    p = _per([{"observable": "x", "value": 3.0, "reference": 1.0}])["x"]
    assert p["direction"] == "over"
    assert "over-predicts" in p["reasoning"]


def test_unverified_value_is_unrated_not_a_failure():
    # An analysis-flagged UNVERIFIED value is untrustworthy evidence, not a
    # confirmed contradiction: unrated, and it must not drive an advisory.
    obs = [{"observable": "viscosity", "value": 0.46, "reference": 2.70,
            "verified": False}]
    assert _per(obs)["viscosity"]["consistent"] is None
    r = run_validation_panel(
        obs, "T1", "sys", judge_fn=numeric_reference_judge,
        advise_fn=lambda flagged, sd: {"recommended_action": "escalate_potential"})
    assert r["failed"] == [] and r["unrated"] == ["viscosity"]
    assert "advisory" not in r          # nothing failed -> no advisory


def test_per_item_tolerance_overrides_default():
    obs = [{"observable": "x", "value": 1.3, "reference": 1.0, "tolerance": 0.5}]
    assert _per(obs)["x"]["consistent"] is True          # 30% within a 50% tol


def test_zero_reference_is_unrated_not_a_pass():
    assert _per([{"observable": "x", "value": 0.1, "reference": 0.0}])["x"]["consistent"] is None


def test_missing_value_is_unrated():
    assert _per([{"observable": "x", "reference": 1.0}])["x"]["consistent"] is None


def test_composes_with_panel_and_advisor():
    # end-to-end deterministic path: a failing observable drives an advisory.
    obs = [{"observable": "viscosity", "value": 0.46, "reference": 2.70}]
    r = run_validation_panel(
        obs, "T1", "some system", judge_fn=numeric_reference_judge,
        advise_fn=lambda flagged, sd: {"recommended_action": "escalate_potential"})
    assert r["failed"] == ["viscosity"]
    assert r["advisory"]["recommended_action"] == "escalate_potential"
