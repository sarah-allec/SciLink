"""ReparameterizationAdvisor recommends how to fix a flagged force field.

Given the inconsistent pure-component properties the critic flagged, it proposes
a concrete corrective action (and whether a human must approve/supply it). The
LLM is stubbed; no API key.
"""

import json
import logging

from scilink.agents.sim_agents.critics import ReparameterizationAdvisor


def _stub(fake_text: str):
    obj = ReparameterizationAdvisor.__new__(ReparameterizationAdvisor)
    obj.logger = logging.getLogger("test_reparam_advisor")
    obj.futurehouse_api_key = None
    captured = {}

    class _Model:
        def generate_content(self, prompt, generation_config=None):
            captured["prompt"] = prompt

            class _Resp:
                text = fake_text

            return _Resp()

    obj.model = _Model()
    return obj, captured


FLAGGED = [
    {"component": "EIS", "property": "density", "consistent": False,
     "reasoning": "Far below the known density of a sulfone."},
]


def test_flagged_properties_and_backend_reach_the_prompt():
    adv, captured = _stub(json.dumps({"recommended_action": "add_force_field"}))
    adv.advise(FLAGGED, system_description="aqueous sulfone electrolyte",
               backend="openff")
    prompt = captured["prompt"]
    assert "EIS" in prompt and "density" in prompt
    assert "Far below the known density" in prompt        # the critic's reasoning
    assert "openff" in prompt
    # The system drives the search itself — the finding is not punted to a human.
    assert "searches the literature" in prompt
    assert "punt the SEARCH to a human" in prompt


def test_signed_numeric_comparison_reaches_the_prompt():
    # A judged observable carries value/reference/direction; the advisor must
    # surface them so the diagnosis reasons FROM the sign, not a textbook default.
    adv, captured = _stub(json.dumps({"recommended_action": "escalate_potential",
                                      "suggested_method": "polarizable_ff"}))
    flagged = [{"observable": "viscosity", "consistent": False,
                "value": 0.74, "reference": 2.7, "units": "mPa*s",
                "direction": "under",
                "reasoning": "computed 0.74 mPa*s vs reference 2.7 mPa*s: model "
                             "under-predicts by 72.7% (exceeds tolerance 25%)"}]
    adv.advise(flagged, system_description="concentrated electrolyte", backend="")
    prompt = captured["prompt"]
    assert "0.74" in prompt and "2.7" in prompt         # the actual numbers
    assert "under-predicts" in prompt                    # the signed direction
    # and the prompt instructs the model to respect that direction
    assert "SIGNED discrepancy" in prompt
    assert "over-prediction mechanism" in prompt


def test_recommendation_passes_through():
    adv, _ = _stub(json.dumps({
        "status": "success",
        "diagnosis": "EIS density is low; the sulfone vdW/charges are under-parameterized.",
        "recommended_action": "add_force_field",
        "detail": "Supplement EIS with a validated sulfone parameter set via extra_force_fields.",
        "requires_human": True,
        "rationale": "Base Sage does not cover sulfones well.",
    }))
    rec = adv.advise(FLAGGED, backend="openff")
    assert rec["recommended_action"] == "add_force_field"
    assert rec["requires_human"] is True
    assert "extra_force_fields" in rec["detail"]


def test_defaults_when_model_underspecifies():
    # Model returns only a diagnosis → advisor fills safe defaults (escalate,
    # human-required) rather than emit an actionable-looking blank.
    adv, _ = _stub(json.dumps({"diagnosis": "unclear"}))
    rec = adv.advise(FLAGGED)
    assert rec["recommended_action"] == "escalate"
    assert rec["requires_human"] is True


def test_no_flagged_is_a_noop_escalate_without_llm():
    adv, captured = _stub("SHOULD NOT BE CALLED")
    rec = adv.advise([])
    assert rec["recommended_action"] == "escalate"
    assert rec["suggested_method"] is None
    assert "prompt" not in captured                        # no LLM call


def test_escalate_potential_and_suggested_method_pass_through():
    # A whole-method-class failure: the advisor escalates to a higher-fidelity
    # potential (not a reparameterization) and names the method.
    adv, _ = _stub(json.dumps({
        "recommended_action": "escalate_potential",
        "suggested_method": "mlip",
        "diagnosis": "transport off by ~5x; classical FF too fluid",
        "detail": "re-run with a pretrained MLIP",
        "rationale": "no per-component parameter change can fix a whole-class error",
    }))
    rec = adv.advise(FLAGGED, backend="openff")
    assert rec["recommended_action"] == "escalate_potential"
    assert rec["suggested_method"] == "mlip"


def test_suggested_method_defaults_none_when_omitted():
    adv, _ = _stub(json.dumps({"recommended_action": "add_force_field"}))
    rec = adv.advise(FLAGGED)
    assert rec["suggested_method"] is None


def test_prompt_offers_escalate_potential_and_generic_mlip():
    adv, captured = _stub(json.dumps({"recommended_action": "escalate_potential",
                                      "suggested_method": "mlip"}))
    adv.advise(FLAGGED, backend="openff")
    prompt = captured["prompt"]
    assert "escalate_potential" in prompt
    assert "machine-learning interatomic potential" in prompt   # generic, not a use case
    assert "suggested_method" in prompt
