"""Observable- and system-aware FF method selection.

The force-field agent, given the target observable in the research goal + the
backend skill's guidance, fills method options the inputs alone don't imply (a
viscosity-accurate water model, ECC charge scaling) — explicit caller values
win, backends without the knob are untouched, and no model is a no-op. LLM
stubbed; no API key.
"""

import logging

from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent


def _agent():
    a = ForceFieldAgent.__new__(ForceFieldAgent)
    a.logger = logging.getLogger("test_ff_method_selection")
    return a


def _openff_like_build(*, components=None, coordinates_file=None, pdb_file=None,
                       working_dir=None, research_goal="", extra_force_fields=None,
                       charge_scaling=None, **_ig):
    return {"extra_force_fields": extra_force_fields, "charge_scaling": charge_scaling}


def _amber_like_build(*, components=None, coordinates_file=None, pdb_file=None,
                      working_dir=None, research_goal="", **_ig):
    return {}


def test_recommendation_injected_when_unset():
    a = _agent()
    a._recommend_method_options = lambda **k: {
        "extra_force_fields": ["opc.offxml"], "charge_scaling": 0.75,
        "rationale": "transport target, concentrated electrolyte"}
    ff = a._apply_method_selection(
        _openff_like_build, backend="openff",
        components=[{"name": "water", "count": 1}],
        research_goal="compute the shear viscosity", ff_kwargs={})
    assert ff["charge_scaling"] == 0.75
    assert ff["extra_force_fields"] == ["opc.offxml"]


def test_explicit_caller_values_win():
    a = _agent()
    a._recommend_method_options = lambda **k: {
        "extra_force_fields": ["opc.offxml"], "charge_scaling": 0.75, "rationale": "x"}
    ff = a._apply_method_selection(
        _openff_like_build, backend="openff", components=[],
        research_goal="viscosity",
        ff_kwargs={"charge_scaling": 0.9, "extra_force_fields": ["tip4p_ew.offxml"]})
    assert ff["charge_scaling"] == 0.9                    # explicit not overridden
    assert ff["extra_force_fields"] == ["tip4p_ew.offxml", "opc.offxml"]  # merged


def test_backend_without_knobs_is_untouched_and_not_queried():
    a = _agent()
    called = {"n": 0}

    def rec(**k):
        called["n"] += 1
        return {"extra_force_fields": ["x"], "charge_scaling": 0.75, "rationale": "y"}

    a._recommend_method_options = rec
    ff = a._apply_method_selection(
        _amber_like_build, backend="amber", components=[],
        research_goal="viscosity", ff_kwargs={"pdb_file": "p.pdb"})
    assert ff == {"pdb_file": "p.pdb"} and called["n"] == 0


def test_no_model_recommendation_is_noop():
    a = _agent()

    def boom(prompt):
        raise RuntimeError("no model available")

    a._generate_json = boom
    rec = a._recommend_method_options(
        backend="openff", components=[{"name": "water", "count": 1}],
        research_goal="viscosity")
    assert rec == {"extra_force_fields": [], "charge_scaling": None, "rationale": ""}


def test_recommend_parses_llm_choice():
    a = _agent()
    a._generate_json = lambda prompt: {
        "water_model": "opc.offxml", "charge_scaling": 0.75,
        "rationale": "transport observable of a concentrated multivalent electrolyte"}
    rec = a._recommend_method_options(
        backend="openff",
        components=[{"name": "water", "count": 500}, {"name": "Zn", "count": 9}],
        research_goal="compute shear viscosity vs composition")
    assert rec["extra_force_fields"] == ["opc.offxml"]
    assert rec["charge_scaling"] == 0.75
    assert rec["rationale"]


def test_recommend_static_target_returns_nothing():
    a = _agent()
    a._generate_json = lambda prompt: {
        "water_model": None, "charge_scaling": None,
        "rationale": "structural target; defaults suffice"}
    rec = a._recommend_method_options(
        backend="openff", components=[{"name": "water", "count": 500}],
        research_goal="compute the density")
    assert rec["extra_force_fields"] == [] and rec["charge_scaling"] is None
