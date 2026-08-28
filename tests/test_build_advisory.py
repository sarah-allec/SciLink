"""build_advisory shapes an advisor recommendation into a human-facing advisory
that is never auto-run. Pure function — no LLM, no engine."""

from scilink.agents.sim_agents.reference_validation import build_advisory


def test_escalate_potential_mlip_names_the_next_step():
    adv = build_advisory({
        "recommended_action": "escalate_potential",
        "suggested_method": "mlip",
        "diagnosis": "transport off ~5x",
        "detail": "re-run with a pretrained MLIP",
        "rationale": "whole-class error",
    })
    assert adv["status"] == "advise_method_escalation"
    assert adv["auto_run"] is False
    assert adv["requires_human_approval"] is True
    assert adv["suggested_method"] == "mlip"
    step = adv["suggested_next_step"]
    assert step["agent"] == "MLIPAgent" and step["method"] == "deploy_pretrained"
    assert "deploy_pretrained" in step["hint"] and "human approval" in step["hint"]


def test_polarizable_ff_escalation_has_no_mlip_step():
    adv = build_advisory({"recommended_action": "escalate_potential",
                          "suggested_method": "polarizable_ff"})
    assert adv["status"] == "advise_method_escalation"
    assert adv["auto_run"] is False
    assert "suggested_next_step" not in adv        # MLIP hint only for mlip


def test_reparameterization_recommendation_is_not_an_escalation():
    adv = build_advisory({"recommended_action": "add_force_field",
                          "suggested_method": None})
    assert adv["status"] == "advise_reparameterization"
    assert adv["auto_run"] is False
    assert "suggested_next_step" not in adv
