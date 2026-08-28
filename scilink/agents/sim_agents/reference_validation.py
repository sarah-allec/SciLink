"""Engine-neutral reference-property validation (pre-run).

Before trusting a novel prediction, validate the model against independently-
known reference quantities of the system's constituents — the "validate before
you predict" discipline (see
``docs/proposals/reference-property-validation-design.md``).

This module is the engine- and force-field-neutral stage. It walks the DISTINCT
components of a system and asks a supplied ``measure_fn`` for each one's
reference property, then assembles a report the physics critic reasons over. It
names no force field and no engine: ``measure_fn`` does the heavy work through
the existing ``ParameterizedSystem`` contract (parameterize -> write_md_inputs
-> run -> read the property), so adding a new backend or engine needs no change
here.
"""

from typing import Any, Callable, Dict, List, Optional


def _component_key(component: Dict[str, Any]) -> str:
    """Identity of a component for de-duplication: SMILES if present, else name."""
    return str(component.get("smiles") or component.get("name") or "").strip()


def validate_component_properties(
    components: List[Dict[str, Any]],
    measure_fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    *,
    reference_property: str = "density",
) -> Dict[str, Any]:
    """Measure each distinct component's reference property for the critic.

    Args:
        components: ``[{"name", "smiles", "count"}, ...]`` — the system's
            component manifest. Repeated species (same SMILES/name) are
            measured once.
        measure_fn: Backend-agnostic measurement of ONE component, returning
            ``{"value": float, "units": str, ...}`` or ``None`` if it could not
            measure. May raise. A component that raises or returns no value is
            recorded as unmeasured, never fatal — partial evidence still helps
            the critic reason.
        reference_property: What ``measure_fn`` measures (default ``"density"``).

    Returns:
        ``{"status", "reference_property", "measurements": [...]}``. Each
        measurement is ``{"component", "smiles", "status", ...}`` with
        ``status`` either ``"measured"`` (carrying ``value``/``units`` plus any
        extra fields the measurer recorded) or ``"unmeasured"`` (carrying
        ``error``). Top-level ``status`` is ``"success"`` if any component was
        measured, else ``"no_measurements"``.
    """
    seen = set()
    measurements: List[Dict[str, Any]] = []
    for comp in components or []:
        key = _component_key(comp)
        if not key or key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {
            "component": comp.get("name") or key,
            "smiles": comp.get("smiles"),
        }
        try:
            result = measure_fn(comp)
        except Exception as e:  # a measurement failure must not sink the stage
            entry.update(status="unmeasured", error=str(e))
            measurements.append(entry)
            continue
        if not result or result.get("value") is None:
            entry.update(status="unmeasured",
                         error=(result or {}).get("error", "no value returned"))
        else:
            entry.update(status="measured", value=result["value"],
                         units=result.get("units"))
            for k, v in result.items():  # carry any extra evidence recorded
                if k not in ("value", "units", "error"):
                    entry[k] = v
        measurements.append(entry)

    any_measured = any(m["status"] == "measured" for m in measurements)
    return {
        "status": "success" if any_measured else "no_measurements",
        "reference_property": reference_property,
        "measurements": measurements,
    }


def run_reference_check(
    components: List[Dict[str, Any]],
    system_description: str,
    *,
    select_fn: Callable[[List[Dict[str, Any]], str], Dict[str, Any]],
    measure_fn: Callable[[Dict[str, Any], str], Optional[Dict[str, Any]]],
    judge_fn: Callable[[List[Dict[str, Any]], str], Dict[str, Any]],
) -> Dict[str, Any]:
    """Pre-run force-field validation: the whole pre-production check in one call.

    Composes the three reasoning/measurement steps so a caller (the
    parameterization gate) runs one thing before committing to production:

    1. ``select_fn`` chooses a reference property per component (density for a
       liquid, a lattice constant for a crystal, ...);
    2. each measurable component's chosen property is measured
       (``measure_fn(component, property)``), collected via
       :func:`validate_component_properties` (dedupe + fail-open);
    3. ``judge_fn`` reasons over the measurements and returns the verdict —
       ``good``, or ``poor`` with the miscalibrated model named.

    All three are injected, so this stays engine/backend-neutral and unit-
    testable: ``select_fn`` / ``judge_fn`` are the reference-property selector /
    critic, ``measure_fn`` the (backend-agnostic) measurement.

    Returns ``{"selections", "status", "reference_property", "measurements",
    "verdict"}`` — the selections, the per-measurement collection, and the
    verdict. A verdict of ``poor`` is the pre-run catch: the force field is
    untrustworthy and production should not proceed unfixed.
    """
    selection = select_fn(components, system_description)

    def _norm(s) -> str:
        return str(s or "").strip().casefold()

    # Index selections by BOTH a normalized name and the SMILES, so a cosmetic
    # mismatch ("EIS" vs "EIS (ethyl isopropyl sulfone)") does not silently drop
    # a component to unmeasured (which would fail the gate open).
    by_name: Dict[str, Any] = {}
    by_smiles: Dict[str, Any] = {}
    for s in selection.get("selections", []):
        if s.get("component"):
            by_name[_norm(s.get("component"))] = s
        if s.get("smiles"):
            by_smiles[_norm(s.get("smiles"))] = s

    def _measure_selected(component: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        chosen = (by_name.get(_norm(component.get("name")))
                  or by_smiles.get(_norm(component.get("smiles"))) or {})
        if not chosen.get("measurable"):
            return {"error": chosen.get("rationale",
                                        "no reference property selected")}
        return measure_fn(component, chosen.get("property"))

    report = validate_component_properties(components, _measure_selected)
    verdict = judge_fn(
        [m for m in report["measurements"] if m.get("status") == "measured"],
        system_description,
    )
    return {"selections": selection.get("selections", []), **report,
            "verdict": verdict}


def _observable_name(entry: Dict[str, Any]) -> str:
    return str(entry.get("observable") or entry.get("component")
               or entry.get("property") or "observable")


def _is_failure(verdict: Dict[str, Any]) -> bool:
    """A per-observable verdict counts as a validation failure."""
    return (verdict.get("verdict") in ("poor", "needs_fixes")
            or verdict.get("consistent") is False)


def _is_rated(verdict: Dict[str, Any]) -> bool:
    """The judge actually rated this observable (has a verdict/consistent field).

    An unrated entry is neither pass nor fail — it is missing evidence, and must
    not be cited as validation.
    """
    return verdict.get("verdict") is not None or verdict.get("consistent") is not None


def _default_scope(prediction_target: str, passed: List[str],
                   failed: List[str], unrated: List[str]) -> str:
    """Deterministic confidence statement scoping the prediction to what passed."""
    passed_str = ", ".join(passed) or "(none)"
    if not failed and not unrated:
        return (f"Prediction of {prediction_target} is backed by validation "
                f"against: {passed_str}.")
    problems = []
    if failed:
        problems.append(f"failed {', '.join(failed)}")
    if unrated:
        problems.append(f"could not rate {', '.join(unrated)}")
    return (f"Prediction of {prediction_target} is NOT fully backed: the model "
            f"{'; '.join(problems)} (validated only {passed_str}). Scope the "
            f"{prediction_target} claim down or fix the model before trusting it.")


def run_validation_panel(
    observations: List[Dict[str, Any]],
    prediction_target: str,
    system_description: str,
    *,
    judge_fn: Callable[[List[Dict[str, Any]], str], Dict[str, Any]],
    scope_fn: Optional[Callable[[str, List[str], List[str], str], str]] = None,
    advise_fn: Optional[Callable[[List[Dict[str, Any]], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate a set of observables against their references; scope the prediction.

    The post-run generalization of :func:`run_reference_check`: instead of one
    constituent property, it judges N observables of the *actual* system (each a
    computed value plus the reference it is judged against — measured data
    preferred, literature as caveated fallback) and aggregates them into a
    confidence statement tied to the ``prediction_target`` (the quantity the run
    predicts, which is NOT itself in ``observations``). A model that reproduces
    several independent observables earns a credible prediction of a correlated
    one; a failure scopes the claim down or routes to the fixer.

    Args:
        observations: ``[{"observable", "value", "units", "reference", ...}, ...]``
            — one computed value per validation observable, with its reference.
        prediction_target: The quantity the validated model will predict.
        system_description: What is being simulated.
        judge_fn: Judges the observations against their references and returns
            ``{"per_observable": [{"observable", "verdict", ...}]}`` (a
            ``per_measurement`` key is also accepted). Injected so the panel stays
            engine/critic-neutral — wire it to the reference-property critic.
        scope_fn: Optional richer (e.g. LLM) confidence-statement generator
            ``(target, passed, failed, system_description) -> str``. Falls back
            to a deterministic summary.
        advise_fn: Optional advisor ``(failed_observables, system_description) ->
            advisory_dict`` — when any observable fails, its result is attached as
            ``advisory`` (e.g. a reparameterization or method-escalation
            recommendation). Injected so the panel stays critic-neutral; wire it
            to the reparameterization advisor. When omitted, no ``advisory`` key
            is added (backward compatible).

    Returns:
        ``{"status", "prediction_target", "per_observable", "passed", "failed",
        "prediction_warranted", "confidence"[, "advisory"]}``.
        ``prediction_warranted`` is True only when no observable failed; ``failed``
        is the catch. ``advisory`` is present only when ``advise_fn`` is given and
        something failed — route those to the fixer / escalate before trusting the
        prediction.
    """
    report = judge_fn(observations, system_description) or {}
    per = report.get("per_observable") or report.get("per_measurement") or []
    failed = [p for p in per if _is_failure(p)]
    passed = [p for p in per if _is_rated(p) and not _is_failure(p)]
    unrated = [p for p in per if not _is_rated(p) and not _is_failure(p)]
    passed_names = [_observable_name(p) for p in passed]
    failed_names = [_observable_name(p) for p in failed]
    unrated_names = [_observable_name(p) for p in unrated]
    # Warranted only when everything was rated AND nothing failed — an unrated
    # observable is missing evidence, not a pass.
    warranted = not failed and not unrated
    if scope_fn is not None:
        confidence = scope_fn(prediction_target, passed_names, failed_names,
                              system_description)
    else:
        confidence = _default_scope(prediction_target, passed_names,
                                    failed_names, unrated_names)
    result = {
        "status": "success",
        "prediction_target": prediction_target,
        "per_observable": per,
        "passed": passed_names,
        "failed": failed_names,
        "unrated": unrated_names,
        "prediction_warranted": warranted,
        "confidence": confidence,
    }
    # On any failure, ask the advisor for a corrective/escalation recommendation
    # (reparameterize a component, or escalate the potential to a higher-fidelity
    # method). Advisory only — never applied here.
    if advise_fn is not None and failed:
        result["advisory"] = advise_fn(failed, system_description)
    return result


def build_advisory(recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """Shape an advisor recommendation into a human-facing advisory.

    The advisory is NEVER auto-run: it carries ``auto_run=False`` and
    ``requires_human_approval=True``. When the recommendation escalates to a
    machine-learning interatomic potential, it also names the concrete (advisory)
    next step; a caller/UI executes it only after a human approves. Engine- and
    potential-neutral; the specific method comes from the injected advisor.
    """
    action = recommendation.get("recommended_action")
    method = recommendation.get("suggested_method")
    advisory = {
        "status": ("advise_method_escalation" if action == "escalate_potential"
                   else "advise_reparameterization"),
        "recommended_action": action,
        "suggested_method": method,
        "diagnosis": recommendation.get("diagnosis"),
        "detail": recommendation.get("detail"),
        "rationale": recommendation.get("rationale"),
        "requires_human_approval": True,
        "auto_run": False,
    }
    if method == "mlip":
        advisory["suggested_next_step"] = {
            "agent": "MLIPAgent",
            "method": "deploy_pretrained",
            "hint": ("MLIPAgent.deploy_pretrained(system_info=..., "
                     "research_goal=..., structure_file=..., "
                     "backend=<mlip backend, e.g. 'mace'>) — run only after "
                     "human approval"),
        }
    return advisory


def numeric_reference_judge(
    observations: List[Dict[str, Any]],
    system_description: str = "",
    *,
    default_tol: float = 0.15,
) -> Dict[str, Any]:
    """A deterministic ``judge_fn`` for :func:`run_validation_panel` (no LLM).

    Flags each observation whose computed ``value`` deviates from its
    ``reference`` by more than a relative tolerance (a per-item ``tolerance``
    overrides ``default_tol``). General — it judges any numeric observable against
    a caller-supplied reference. A ``direction`` (``"over"``/``"under"``) records
    whether the model over- or under-predicts, so a downstream diagnosis reasons
    from the SIGNED discrepancy rather than assuming a failure mode. Observations
    missing a numeric value/reference — or explicitly carrying ``verified=False``
    (the value the analysis could not trust) — are left unrated
    (``consistent=None``), never a false pass and never the sole basis of an
    advisory. Returns ``{"per_observable": [{observable, consistent, rel_error?,
    direction?, value, reference, units, reasoning}]}``.
    """
    per: List[Dict[str, Any]] = []
    for o in observations:
        name = o.get("observable") or o.get("name") or "(observable)"
        entry: Dict[str, Any] = {
            "observable": name, "value": o.get("value"),
            "reference": o.get("reference"), "units": o.get("units"),
        }
        # An UNVERIFIED value is untrustworthy evidence, not a confirmed
        # contradiction — leave it unrated so it neither passes nor drives an
        # advisory off a number the analysis itself flagged as implausible.
        if o.get("verified") is False:
            entry.update(consistent=None,
                         reasoning="computed value is unverified; not rated")
            per.append(entry)
            continue
        try:
            v = float(o.get("value"))
            r = float(o.get("reference"))
        except (TypeError, ValueError):
            entry.update(consistent=None,
                         reasoning="missing or non-numeric value/reference")
            per.append(entry)
            continue
        if r == 0:
            entry.update(consistent=None,
                         reasoning="reference is zero; relative error undefined")
        else:
            tol = float(o.get("tolerance", default_tol))
            rel = abs(v - r) / abs(r)
            direction = "over" if v > r else "under" if v < r else "exact"
            units = o.get("units")
            us = f" {units}" if units else ""
            within = "within" if rel <= tol else "exceeds"
            if direction == "exact":
                reasoning = (f"computed {v:.4g}{us} matches reference {r:.4g}{us} "
                             f"(0%, {within} tolerance {tol:.0%})")
            else:
                reasoning = (f"computed {v:.4g}{us} vs reference {r:.4g}{us}: model "
                             f"{direction}-predicts by {rel:.1%} "
                             f"({within} tolerance {tol:.0%})")
            entry.update(consistent=bool(rel <= tol), rel_error=round(rel, 4),
                         direction=direction, reasoning=reasoning)
        per.append(entry)
    return {"per_observable": per}


def run_reparameterization(
    flagged: List[Dict[str, Any]],
    system_description: str,
    backend: str,
    *,
    advise_fn: Callable[[List[Dict[str, Any]], str, str], Dict[str, Any]],
    search_fn: Callable[[Dict[str, Any], List[Any]], Optional[Any]],
    apply_and_recheck_fn: Callable[[Any], Dict[str, Any]],
    confirm_fn: Callable[[Any], bool] = lambda candidate: True,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """Autonomously fix a force field the pre-run check flagged, and re-validate.

    SciLink drives the fix; the human only approves. The loop:

    1. ``advise_fn`` recommends a corrective action for the flagged properties;
    2. ``search_fn`` finds a candidate correction (e.g. literature parameters
       for the offending component), given what has already been tried;
    3. ``confirm_fn`` is the human checkpoint (approve the candidate) — the
       identity default auto-approves for autonomous runs;
    4. ``apply_and_recheck_fn`` applies the candidate (re-parameterizes) and
       re-runs the pure-component check — the SAME check that caught the problem
       now validates the fix, so a wrong candidate fails here and is discarded.

    Repeats up to ``max_attempts`` distinct candidates. All operations are
    injected, so this is engine/backend-neutral and unit-testable without a
    model, a literature search, or a simulation.

    Returns ``{"status", "recommendation", "candidate"?, "reference_validation"?,
    "attempts"}`` where ``status`` is:
    ``"fixed"`` (a candidate re-validated), ``"advise_method_escalation"`` (the
    method class is inadequate — escalate the potential, human-approved, not run
    here), ``"escalated"`` (no automatic action to attempt), ``"no_candidate"``
    (search found nothing), ``"declined"`` (human rejected a candidate), or
    ``"unresolved"`` (candidates tried, none passed).
    """
    recommendation = advise_fn(flagged, system_description, backend)
    action = recommendation.get("recommended_action")
    if action == "escalate_potential":
        # Method-class inadequacy: no per-component parameter fix applies. Surface
        # the escalation recommendation for a human to approve; do NOT enter the
        # parameter search/apply loop.
        return {"status": "advise_method_escalation",
                "recommendation": recommendation, "attempts": []}
    if action in (None, "escalate"):
        return {"status": "escalated", "recommendation": recommendation,
                "attempts": []}

    tried: List[Any] = []
    attempts: List[Dict[str, Any]] = []
    for _ in range(max(1, max_attempts)):
        candidate = search_fn(recommendation, tried)
        if not candidate:
            return {"status": "no_candidate" if not tried else "unresolved",
                    "recommendation": recommendation, "attempts": attempts}
        if not confirm_fn(candidate):
            return {"status": "declined", "recommendation": recommendation,
                    "candidate": candidate, "attempts": attempts}
        result = apply_and_recheck_fn(candidate)
        verdict = (result.get("verdict") or {}).get("verdict")
        attempts.append({"candidate": candidate, "verdict": verdict})
        if verdict == "good":
            return {"status": "fixed", "recommendation": recommendation,
                    "candidate": candidate, "reference_validation": result,
                    "attempts": attempts}
        tried.append(candidate)

    return {"status": "unresolved", "recommendation": recommendation,
            "attempts": attempts}
