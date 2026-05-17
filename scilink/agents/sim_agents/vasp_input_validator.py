"""
VASP-specific pre-run input validator.

Two-layer pre-run validation for VASP INCAR files:

  1. Engine-native syntax check  (``check_incar_syntax``)
     Wraps pymatgen's ``Incar.check_params()`` to catch typoed /
     non-conforming INCAR tags (e.g. ``ISPN = 2`` instead of ``ISPIN``).
     No LLM call.  Returns structured issues with high/low-confidence
     rename suggestions derived from difflib against the canonical
     VASP tag list.

  2. LLM / literature check       (``IncarValidatorAgent.validate_and_improve_incar``)
     Asks a FutureHouse literature agent + an LLM to judge whether the
     parameter choices are physically appropriate for the system.
     Pre-existing behavior — moved here from val_agent.py as part of
     the 2026-05-17 split.

Why two layers:  VASP accepts unknown INCAR keys silently — it just
emits a one-line OUTCAR warning and runs with that key ignored.  An
``ISPN`` typo can therefore disable spin polarisation on Fe and produce
a physics-wrong result that converges by every other metric.  The
syntax layer catches this before submission; the LLM layer catches
"valid tag, wrong value for this system".

Engine-neutral consumers (orchestrators, benchmark harness, future
meta-agent) call ``check_incar_syntax`` / ``apply_incar_syntax_fixes``
and never import pymatgen directly.  pymatgen import is lazy so a
machine without it returns an empty issue list rather than crashing.

The shape ``check_syntax(content) -> List[issue]`` + optional
``validate_and_improve()`` is the engine-neutral contract.  Future
``lammps_input_validator.py`` and ``gromacs_input_validator.py``
should mirror it; see CLAUDE.md "Engine-neutral contracts".
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings as _warnings
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Dict, List, Optional, Tuple

from ...auth import (
    APIKeyNotFoundError, get_api_key, get_internal_proxy_key, infer_provider,
)
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ..lit_agents.literature_agent import IncarLiteratureAgent
from ._deprecation import normalize_params
from .instruct import INCAR_VALIDATION_INSTRUCTIONS


# Tokens that show up inside legitimate INCAR *values* (e.g. ``LDAUL = 3
# 3 -1``, ``LSORBIT = .TRUE.``) rather than as standalone tags.  Filtered
# out of the difflib suggestion pool so we don't propose nonsense
# renames.
_VASP_SUGGESTION_BLOCKLIST = {"TRUE", "FALSE"}


def _load_valid_vasp_tags() -> List[str]:
    """Return the canonical VASP INCAR-tag list bundled with pymatgen.

    Returns an empty list if pymatgen is unavailable or the bundled JSON
    has moved — callers degrade gracefully (no rename suggestions).
    """
    try:
        from pymatgen.io.vasp import inputs as _vasp_inputs
    except Exception:
        return []
    tags_path = os.path.join(
        os.path.dirname(_vasp_inputs.__file__), "incar_parameters.json"
    )
    if not os.path.exists(tags_path):
        return []
    try:
        with open(tags_path) as f:
            return list(json.load(f).keys())
    except Exception:
        return []


def check_incar_syntax(incar_content: str) -> List[Dict[str, Any]]:
    """Engine-native pre-run syntax check for a VASP INCAR.

    Parameters
    ----------
    incar_content : str
        Raw INCAR text (not a path).  Pass ``open(...).read()``.

    Returns
    -------
    list of issue dicts (possibly empty).  Each dict carries:

        severity    : "warning"
        category    : "incar_tag"
        tag         : the offending tag as written (str or None)
        suggested   : closest valid VASP tag (str or None)
        confidence  : "high" | "low"
        description : human-readable summary
        source      : "pymatgen Incar.check_params"

    ``confidence="high"`` requires the top match to be ≥0.85 similar to
    the bad tag AND clearly better than the runner-up (>0.05 margin).
    The auto-fix path (``apply_incar_syntax_fixes``) only consumes
    high-confidence entries; low-confidence entries are returned for the
    LLM regenerator to consider as context.
    """
    try:
        from pymatgen.io.vasp.inputs import Incar
    except Exception:
        return []

    try:
        if hasattr(Incar, "from_str"):
            incar = Incar.from_str(incar_content)
        else:
            incar = Incar.from_string(incar_content)  # older pymatgen
    except Exception:
        # Malformed INCAR — let VASP itself complain.  Syntax pass is
        # specifically for the "syntactically valid but contains a fake
        # tag" failure mode.
        return []

    valid_tags = [
        t for t in _load_valid_vasp_tags()
        if t not in _VASP_SUGGESTION_BLOCKLIST
    ]

    issues: List[Dict[str, Any]] = []
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            incar.check_params()
        except Exception:
            return []

    for w in caught:
        msg = str(w.message)
        m = re.search(r"Cannot find\s+(\S+)", msg)
        bad_tag = m.group(1).strip().strip(",.") if m else None

        suggested: Optional[str] = None
        confidence = "low"
        if bad_tag and valid_tags:
            matches = get_close_matches(
                bad_tag.upper(), valid_tags, n=2, cutoff=0.7
            )
            if matches:
                suggested = matches[0]
                top_sim = SequenceMatcher(
                    None, bad_tag.upper(), matches[0]
                ).ratio()
                runner_sim = (
                    SequenceMatcher(None, bad_tag.upper(), matches[1]).ratio()
                    if len(matches) > 1 else 0.0
                )
                if top_sim >= 0.85 and (top_sim - runner_sim) >= 0.05:
                    confidence = "high"

        issues.append({
            "severity": "warning",
            "category": "incar_tag",
            "tag": bad_tag,
            "suggested": suggested,
            "confidence": confidence,
            "description": (
                f"INCAR tag '{bad_tag}' is not recognised by VASP. "
                f"Closest match: {suggested}."
                if bad_tag and suggested else msg
            ),
            "source": "pymatgen Incar.check_params",
        })

    return issues


def apply_incar_syntax_fixes(
    incar_content: str,
    issues: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply high-confidence tag renames in-place to an INCAR string.

    Parameters
    ----------
    incar_content : str
        Raw INCAR text.
    issues : list, optional
        Issues from a prior ``check_incar_syntax`` call.  If omitted,
        the check is run inline.

    Returns
    -------
    (fixed_content, applied) : tuple
        ``fixed_content`` is the (possibly modified) INCAR text.
        ``applied`` is the subset of issues whose rename actually fired,
        each augmented with ``renamed_from`` / ``renamed_to`` keys.

    Low-confidence issues are returned untouched; caller is expected to
    forward them to the LLM regenerator as context.
    """
    if issues is None:
        issues = check_incar_syntax(incar_content)

    fixed = incar_content
    applied: List[Dict[str, Any]] = []
    for issue in issues:
        if issue.get("category") != "incar_tag":
            continue
        if issue.get("confidence") != "high":
            continue
        bad = issue.get("tag")
        good = issue.get("suggested")
        if not bad or not good:
            continue
        # Only rename when ``bad`` is on the LHS of an assignment.  The
        # word-boundary regex + line anchor avoids touching value tokens
        # that happen to share the spelling.
        pattern = re.compile(rf"(?im)^(\s*){re.escape(bad)}(\s*=)")
        new_fixed, n = pattern.subn(rf"\1{good}\2", fixed)
        if n > 0:
            fixed = new_fixed
            applied.append({**issue, "renamed_from": bad, "renamed_to": good})
    return fixed, applied


class IncarValidatorAgent:
    """Agent that validates and suggests improvements to VASP INCAR files.

    Two surfaces:

      * ``check_syntax(incar_content)`` — engine-native pymatgen check.
        No LLM, no FutureHouse dependency.  Returns the same list shape
        as the module-level ``check_incar_syntax`` and is provided as a
        convenience so callers that already hold a validator instance
        don't need a separate import.

      * ``validate_and_improve_incar(incar_content, system_description)``
        — LLM-driven literature review of parameter choices.  Requires
        an LLM model + FutureHouse API key (existing pre-PR behavior).

    The two surfaces are independent: a caller that only needs the
    syntax pass can use the module function and skip the LLM init
    entirely.
    """

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 futurehouse_api_key: str = None,
                 max_wait_time: int = 500,
                 # Legacy params
                 local_model: str = None,
                 google_api_key: str = None):

        self.logger = logging.getLogger(__name__)

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="IncarValidatorAgent"
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            if api_key is None:
                provider = infer_provider(model_name) or "google"
                api_key = get_api_key(provider) or get_internal_proxy_key()
            if not api_key:
                raise APIKeyNotFoundError(
                    infer_provider(model_name) or "google"
                )
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )

        self.generation_config = None

        self.literature_agent = IncarLiteratureAgent(
            api_key=futurehouse_api_key,
            max_wait_time=max_wait_time
        )

    def check_syntax(self, incar_content: str) -> List[Dict[str, Any]]:
        """Instance-method convenience wrapper around ``check_incar_syntax``."""
        return check_incar_syntax(incar_content)

    def validate_and_improve_incar(self, incar_content: str,
                                   system_description: str) -> dict:
        """Validate INCAR parameters and suggest improvements based on literature."""

        self.logger.info("Getting literature review of INCAR parameters...")
        lit_result = self.literature_agent.validate_incar(
            incar_content, system_description
        )

        if lit_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Literature review failed: {lit_result.get('message')}",
                "validation_status": "unknown"
            }

        self.logger.info("Analyzing literature review for potential improvements...")

        prompt = f"""{INCAR_VALIDATION_INSTRUCTIONS}

## ORIGINAL INCAR:
{incar_content}

## SYSTEM DESCRIPTION:
{system_description}

## LITERATURE REVIEW:
{lit_result['response']}

Analyze the literature review and suggest specific parameter adjustments if needed."""

        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            result = json.loads(response.text)
            result.update({
                "status": "success",
                "literature_review": lit_result['response'],
                "literature_task_id": lit_result.get('task_id')
            })
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing literature review: {e}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}",
                "literature_review": lit_result['response']
            }

    def save_validation_report(self, validation_result: dict,
                               output_dir: str = ".") -> dict:
        """Save validation report and revised INCAR if needed."""
        if validation_result.get("status") != "success":
            return {"error": "Validation was not successful"}

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        try:
            report_path = os.path.join(output_dir, "incar_validation_report.json")
            with open(report_path, 'w') as f:
                json.dump(validation_result, f, indent=2, default=str)
            saved_files["validation_report"] = report_path

            if (validation_result.get("validation_status") == "needs_adjustment" and
                    validation_result.get("revised_incar")):

                revised_path = os.path.join(output_dir, "INCAR_revised")
                with open(revised_path, 'w') as f:
                    f.write(validation_result["revised_incar"])
                saved_files["revised_incar"] = revised_path

                summary_path = os.path.join(output_dir, "incar_adjustments.txt")
                with open(summary_path, 'w') as f:
                    f.write("INCAR Parameter Adjustments\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(f"Overall Assessment: {validation_result.get('overall_assessment', 'N/A')}\n\n")

                    adjustments = validation_result.get("suggested_adjustments", [])
                    if adjustments:
                        f.write("Suggested Changes:\n")
                        for adj in adjustments:
                            f.write(f"\n• {adj.get('parameter')}:\n")
                            f.write(f"  Current: {adj.get('current_value')}\n")
                            f.write(f"  Suggested: {adj.get('suggested_value')}\n")
                            f.write(f"  Reason: {adj.get('reason')}\n")
                    else:
                        f.write("No specific adjustments suggested.\n")

                saved_files["adjustment_summary"] = summary_path

            self.logger.info(f"Validation report saved: {saved_files}")
            return saved_files

        except Exception as e:
            self.logger.error(f"Error saving validation files: {e}")
            return {"error": f"Save failed: {str(e)}"}
