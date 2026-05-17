"""Post-run quality / critic agent for VASP calculations.

Wraps `post_run_analysis.analyze_run_directory` (the deterministic
data layer that parses vasprun.xml + logs) with an LLM layer that
interprets those facts in context of the research goal and the
VASP-skill conventions.

Returns the same structured contract as
`LAMMPSAnalysisAgent.run_quality_check`, so an orchestrator can
dispatch to either engine without branching.

Two layers, mirroring `VaspUpdater`:
  1. Deterministic guardrails (convergence flags, max-force-vs-EDIFFG,
     SCF saturation, classified error patterns) — always applied.
  2. LLM synthesis — physics-aware judgment that catches issues the
     guardrails can't (e.g. low-spin ground state when high-spin was
     expected; final energy outside the typical PBE band).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ...auth import (
    APIKeyNotFoundError,
    get_api_key,
    get_internal_proxy_key,
    infer_provider,
)
from ...skills.loader import load_skill
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ._deprecation import normalize_params
from .instruct import (
    VASP_KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
    VASP_SKILL_UPDATE_INSTRUCTIONS,
)
from .post_run_analysis import analyze_run_directory
from .skill_graduation import (
    KnowledgeStore,
    format_graduated_skills_block,
    graduate_to_skill_file,
    load_graduated_skills,
)


# ──────────────────────────────────────────────────────────────
# Deterministic guardrails
# ──────────────────────────────────────────────────────────────

def _check_incar_tags(incar_path: str) -> List[Dict[str, Any]]:
    """Post-run wrapper around the engine-native INCAR syntax check.

    Reads the INCAR off disk and delegates to
    ``vasp_input_validator.check_incar_syntax``.  The pre-run path
    (PeriodicDFTAgent) usually catches typos before submission; this
    post-run pass exists as a defense-in-depth check so that runs that
    bypassed pre-validation (manual INCAR edits, externally-supplied
    inputs) still get the warning.

    Same return shape as ``check_incar_syntax`` — keeps the issue
    contract consistent between Generate and Quality stages.
    """
    if not os.path.exists(incar_path):
        return []
    try:
        with open(incar_path) as f:
            content = f.read()
    except Exception:
        return []
    from .vasp_input_validator import check_incar_syntax
    return check_incar_syntax(content)


def _deterministic_issues(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Issues that should always be flagged based on the parsed facts.

    Kept independent of LLM judgment so a clearly-broken run is
    flagged even if the LLM call fails or returns garbage.
    """
    issues: List[Dict[str, Any]] = []

    if facts.get("converged_electronic") is False:
        issues.append({
            "severity": "critical",
            "description": "Electronic SCF did not converge to EDIFF.",
            "source": "vasprun.xml convergence flag",
        })

    if facts.get("converged_ionic") is False:
        issues.append({
            "severity": "critical",
            "description": "Ionic relaxation did not reach EDIFFG.",
            "source": "vasprun.xml convergence flag",
        })

    incar = facts.get("incar_snapshot", {}) or {}

    # Forces vs EDIFFG (only meaningful if a relaxation was requested).
    nsw_raw = incar.get("NSW", 0)
    ediffg_raw = incar.get("EDIFFG")
    max_force = facts.get("max_force_eV_per_A")
    try:
        nsw = int(nsw_raw)
    except (TypeError, ValueError):
        nsw = 0
    if nsw > 0 and ediffg_raw is not None and max_force is not None:
        try:
            ediffg = float(ediffg_raw)
            if ediffg < 0:  # negative EDIFFG = force-based threshold
                threshold = abs(ediffg)
                if max_force > threshold:
                    issues.append({
                        "severity": "warning",
                        "description": (
                            f"Max force {max_force:.4f} eV/Å exceeds the "
                            f"EDIFFG threshold {threshold:.4f} eV/Å. "
                            "Relaxation may not be fully converged."
                        ),
                        "source": "max_force vs EDIFFG comparison",
                        "metric": {
                            "max_force_eV_per_A": max_force,
                            "threshold_eV_per_A": threshold,
                        },
                    })
        except (TypeError, ValueError):
            pass

    # SCF iteration saturation on the last ionic step.
    nelm_raw = incar.get("NELM", 60)
    n_scf_last = facts.get("n_electronic_steps_last_ionic")
    try:
        nelm = int(nelm_raw)
    except (TypeError, ValueError):
        nelm = 60
    if (
        nelm
        and isinstance(n_scf_last, (int, float))
        and n_scf_last >= nelm
    ):
        issues.append({
            "severity": "warning",
            "description": (
                f"Last ionic step's electronic SCF reached NELM={nelm}; "
                "the run may be at the iteration ceiling and not fully "
                "converged."
            ),
            "source": "n_electronic_steps_last_ionic vs NELM",
            "metric": {"n_scf_last": n_scf_last, "nelm": nelm},
        })

    # Classified error patterns from log tails (post_run_analysis surfaces
    # these via _classify_log_errors).
    err_hints = facts.get("error_hints") or facts.get("classified_errors") or []
    for hint in err_hints:
        issues.append({
            "severity": "warning",
            "description": f"Log pattern flagged: {hint}",
            "source": "stdout/stderr classification",
        })

    return issues


# ──────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────

class VaspQualityAgent:
    """Two-layer post-run quality assessment for VASP calculations.

    Output contract matches `LAMMPSAnalysisAgent.run_quality_check`:
        {
            "status": "healthy" | "warning" | "critical" | "unknown",
            "can_continue": bool,
            "issues": [{"severity", "description", ...}, ...],
            "recommendations": [str, ...],
            "quality_metrics": {numeric/boolean facts},
            "assessment_summary": str,
            "facts": {raw analyze_run_directory output for reference},
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        # Deprecated aliases (kept for parity with sibling agents)
        local_model: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="VaspQualityAgent",
        )

        # Provider-aware key resolution. Mirrors the fix recently landed
        # in DFTOrchestrator -- infer the provider from the model name,
        # then fall back to the SCILINK_API_KEY proxy.
        if api_key is None and base_url is None:
            provider = infer_provider(model_name) or "google"
            api_key = get_api_key(provider) or get_internal_proxy_key()
            if not api_key:
                raise APIKeyNotFoundError(provider)

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
            )

        self.generation_config = None

        # In-session observations awaiting graduation (e.g. "noticed
        # ALGO=All + ISMEAR=-5 produced tetrahedron warning").
        self.knowledge_store = KnowledgeStore()

    # ── On-the-fly skill graduation ──────────────────────────

    def record_knowledge(self, observation: Dict[str, Any]) -> str:
        """Record a quality-side observation for later graduation.

        Free-form dict; common keys: `summary`, `incar_snapshot`,
        `flagged_combination`, `recommended_fix`. Returns the assigned id."""
        return self.knowledge_store.record(observation)

    def list_knowledge(self) -> List[Dict[str, Any]]:
        return self.knowledge_store.list()

    def clear_knowledge(self, knowledge_id: Optional[str] = None) -> int:
        return self.knowledge_store.remove(knowledge_id)

    def graduate_to_skill(
        self,
        knowledge_id: str,
        skill_name: str = "vasp_quality_heuristics",
    ) -> Dict[str, Any]:
        """Crystallize a quality-side observation into a graduated skill.

        Default skill name is `vasp_quality_heuristics`, which keeps the
        post-run quality rules separate from the input-fix rules
        (`vasp_learned_fixes`). Both end up under
        ~/.scilink/graduated_skills/vasp/, so the loader picks them up
        automatically next session."""
        entry = self.knowledge_store.get(knowledge_id)
        if entry is None:
            return {
                "status": "error",
                "message": f"Knowledge id not found: {knowledge_id}",
            }

        def _llm_call(prompt: str) -> str:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            return response.text

        return graduate_to_skill_file(
            knowledge_entry=entry,
            skill_name=skill_name,
            domain="vasp",
            llm_call=_llm_call,
            fresh_template=VASP_KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
            update_template=VASP_SKILL_UPDATE_INSTRUCTIONS,
        )

    # ── Public entry point ─────────────────────────────────────

    def run_quality_check(
        self,
        output_dir: str,
        research_goal: str,
        skill: Optional[str] = "vasp",
    ) -> Dict[str, Any]:
        """Assess the post-run quality of a VASP calculation.

        Args:
            output_dir: directory containing the VASP outputs (vasprun.xml,
                OUTCAR, OSZICAR, stdout/stderr).
            research_goal: the original natural-language objective for
                the calculation. Provides the LLM with intent context.
            skill: VASP skill name to load convention guidance from
                (defaults to "vasp", which loads
                ``skills/periodic_dft/vasp/vasp.md``; the legacy name
                "vasp_input_generation" is also accepted via the alias
                map). Pass None to skip skill loading.
        """
        self.logger.info(f"Running VASP quality check on: {output_dir}")

        # Layer 1: deterministic facts + guardrails.
        facts = analyze_run_directory(output_dir)
        det_issues = _deterministic_issues(facts)
        # Plus: scan the on-disk INCAR for typoed / unknown tags.
        # Picks up things VASP silently ignores (e.g. ISPN-for-ISPIN).
        det_issues.extend(
            _check_incar_tags(os.path.join(output_dir, "INCAR"))
        )

        # Skill convention text (validation section preferred).
        skill_text = self._load_skill_text(skill)

        # Layer 2: LLM synthesis.
        llm_assessment = self._synthesize(
            facts=facts,
            det_issues=det_issues,
            research_goal=research_goal,
            skill_text=skill_text,
        )

        # Combine: guardrails always present; LLM-found issues appended.
        combined_issues = list(det_issues)
        for issue in llm_assessment.get("issues", []) or []:
            combined_issues.append(issue)

        # Status: any critical guardrail wins; otherwise trust LLM.
        det_severities = {i["severity"] for i in det_issues}
        if "critical" in det_severities:
            status = "critical"
            can_continue = False
        elif llm_assessment.get("status") == "critical":
            status = "critical"
            can_continue = False
        elif "warning" in det_severities or llm_assessment.get("status") == "warning":
            status = "warning"
            can_continue = bool(llm_assessment.get("can_continue", True))
        else:
            status = llm_assessment.get("status", "unknown")
            can_continue = bool(llm_assessment.get("can_continue", True))

        return {
            "status": status,
            "can_continue": can_continue,
            "issues": combined_issues,
            "recommendations": llm_assessment.get("recommendations", []) or [],
            "quality_metrics": self._extract_metrics(facts),
            "assessment_summary": llm_assessment.get(
                "assessment_summary", "Assessment summary unavailable."
            ),
            "facts": facts,
        }

    # ── Internals ──────────────────────────────────────────────

    # Backward-compat: previous skill location was vasp/vasp_input_generation/,
    # now at periodic_dft/vasp/ after the scale-aware refactor. Keep the same
    # alias mapping PeriodicDFTAgent uses so direct-to-loader callers (this
    # method) resolve the legacy name too.
    _LEGACY_SKILL_ALIASES = {"vasp_input_generation": "vasp"}

    def _load_skill_text(self, skill: Optional[str]) -> str:
        if not skill:
            return ""
        resolved = self._LEGACY_SKILL_ALIASES.get(skill, skill)
        try:
            parsed = load_skill(resolved, domain="periodic_dft")
        except Exception as exc:
            self.logger.warning(f"Could not load skill '{resolved}': {exc}")
            return ""
        # Validation section is the most relevant for post-run judgment;
        # fall back to planning if validation is empty.
        return parsed.get("validation") or parsed.get("planning") or ""

    @staticmethod
    def _extract_metrics(facts: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten the numeric / boolean facts into a quality_metrics dict.

        Mirrors what the LAMMPS quality agent surfaces in its
        quality_metrics field — only types that are easy to put in a
        downstream report or table.
        """
        keep = (
            "converged",
            "converged_electronic",
            "converged_ionic",
            "final_energy",
            "n_ionic_steps",
            "n_electronic_steps_last_ionic",
            "max_force_eV_per_A",
        )
        return {k: facts[k] for k in keep if k in facts}

    def _synthesize(
        self,
        *,
        facts: Dict[str, Any],
        det_issues: List[Dict[str, Any]],
        research_goal: str,
        skill_text: str,
    ) -> Dict[str, Any]:
        """LLM call: turn facts + guardrails into a contextual assessment."""
        facts_str = json.dumps(facts, indent=2, default=str)
        det_issues_str = json.dumps(det_issues, indent=2, default=str)
        skill_block = f"\nVASP CONVENTIONS (from skill):\n{skill_text}\n" if skill_text else ""
        # Graduated rules (e.g. "ALGO=All + ISMEAR=-5 is incompatible")
        # — empty string when nothing has been graduated yet.
        learned_block = format_graduated_skills_block(
            load_graduated_skills("vasp"),
        )

        prompt = f"""Assess the post-run quality of a VASP calculation against the research goal.

RESEARCH GOAL:
{research_goal}

FACTS FROM THE RUN (parsed from vasprun.xml + log tails):
{facts_str}

DETERMINISTIC GUARDRAILS ALREADY FLAGGED (do not duplicate; supplement):
{det_issues_str}
{skill_block}{learned_block}
Assess:
1. Overall status — "healthy" (everything looks right), "warning" (minor
   physics concerns), or "critical" (the calculation cannot be trusted).
2. Whether downstream analysis / property computation can proceed
   (can_continue: true/false).
3. Specific physics-aware issues NOT already in the deterministic list.
   Examples: magnetic moment too low for high-spin Fe; final energy
   far from typical PBE values for this system; the agent asked for
   a relaxation but the structure barely moved (n_ionic_steps == 1
   despite NSW > 1); k-point sampling looks too coarse for the cell;
   NSW saturated without reaching EDIFFG (ran out of ionic steps).
4. Concrete actionable recommendations.

Return strictly valid JSON only, no prose outside the JSON object:
{{
    "status": "healthy|warning|critical",
    "can_continue": true,
    "issues": [
        {{"severity": "critical|warning|info", "description": "..."}}
    ],
    "recommendations": ["..."],
    "assessment_summary": "one or two short paragraphs"
}}
"""
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            return self._parse_json(response.text)
        except Exception as exc:
            self.logger.error(f"LLM synthesis failed: {exc}")
            return {
                "status": "unknown",
                "can_continue": True,
                "issues": [],
                "recommendations": [
                    "Manual review recommended -- LLM synthesis failed."
                ],
                "assessment_summary": f"LLM-side assessment failed: {exc}",
            }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Tolerant JSON extraction. Handles bare JSON, fenced ```json
        blocks, and surrounding prose."""
        # Strip Markdown code fences if present.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))
        # Direct parse first.
        try:
            return json.loads(text)
        except Exception:
            pass
        # Last resort: greedy match a {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
