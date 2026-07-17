"""Scale-agnostic deterministic simulation pipeline.

A single one-shot pipeline that turns a natural-language request into a
validated structure plus ready-to-run inputs, for any simulation scale.
One scale-agnostic entry point (``run_complete_workflow``) serves every
engine; the scale selects the foundation agent and the engine selects the
skill bundle.

The pipeline is deterministic — it runs a fixed step sequence rather than
letting an orchestration LLM choose steps — which is what makes its output
reproducible for benchmarking. The chat / LLM-driven path lives on
``SimulationOrchestratorAgent`` (``chat`` / ``run_task``); this is the
headless sequence both that orchestrator and analyze-mode call.

Steps:
    1. Structure   — StructurePipeline (scale-agnostic) builds and
                     validates the atomic structure.
    2. Inputs      — the routed scale's foundation agent generates inputs,
                     returning a normalized ``input_files`` map (engine
                     selected by ``software``; an optional named ``method``
                     selects a deterministic generation backend registered
                     in the engine's skill bundle).
    3. Validation  — InputValidator reviews the generated inputs (skill
                     guidance + deterministic syntax check + literature
                     grounding when a FutureHouse key is present).

Adding a new scale (e.g. molecular DFT) is a new foundation agent plus a
skill bundle and one dispatch branch in ``_generate_inputs`` — no new
orchestrator class, and no hardcoded engine filenames anywhere.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Filesystem-safe slug for a member/run directory name."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")
    return slug or "member"


# Default engine per scale, used when the caller does not name one. Each
# scale's foundation agent resolves the engine to a skill bundle.
_DEFAULT_ENGINE = {
    "periodic_dft": "vasp",
    "molecular_dynamics": "lammps",
}


def _generate_inputs(
    *,
    scale: str,
    software: str,
    method: str,
    structure_file: str,
    request: str,
    output_dir: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
    force_field_files: Optional[Dict[str, str]] = None,
    staged: bool = False,
) -> Dict[str, Any]:
    """Generate inputs for ``scale``, returning a normalized result.

    Every branch returns a result dict carrying an ``input_files`` mapping
    (filename → contents), so downstream steps never guess engine
    filenames. When ``method`` names a deterministic backend (anything
    other than ``"llm"``), inputs come from a ``generate_inputs_<method>``
    tool in the engine's skill bundle; otherwise the routed scale's
    foundation agent produces them with its default LLM path.

    Args:
        scale: Simulation scale (e.g. ``"periodic_dft"``).
        software: Engine name within the scale (e.g. ``"vasp"``).
        method: ``"llm"`` for the agent's baseline generation, or a named
            backend resolved from the skill bundle.
        structure_file: Path to the built structure.
        request: The scientific request driving parameter choices.
        output_dir: Where inputs should be written.
        api_key, base_url, model_name: LLM credentials forwarded to the
            foundation agent.

    Returns:
        A dict with at least ``status`` and, on success, an ``input_files``
        mapping (filename → contents).

    Raises:
        ValueError: If the scale is not supported by the pipeline.
    """
    # Named deterministic backend: a skill-bundle generation tool. The tool
    # is responsible for returning a normalized input_files map.
    if method and method != "llm":
        from ...skills._shared._registry import get_tool_function
        gen = get_tool_function(f"generate_inputs_{method}", active_skills=[software])
        return gen(structure_file=structure_file, request=request,
                   output_dir=output_dir)

    if scale == "periodic_dft":
        from .periodic_dft_agent import PeriodicDFTAgent
        agent = PeriodicDFTAgent(
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        result = agent.generate_inputs(
            structure_file=structure_file, request=request, software=software,
        )
        # PeriodicDFTAgent already returns input_files as {filename: contents}.
        if result.get("status") == "success":
            agent.save_inputs(result, output_dir)
        return result

    if scale == "molecular_dynamics":
        from .md_simulation_agent import MDSimulationAgent
        agent = MDSimulationAgent(
            working_dir=output_dir,
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        # Staged generation emits an optimization → equilibration → production
        # chain as a normalized sequential campaign; one-shot generation emits a
        # single phase (or a parallel sweep when the plan calls for one). Both
        # return the same normalized result shape the pipeline consumes.
        gen = (agent.generate_staged_simulation if staged
               else agent.generate_simulation)
        result = gen(
            structure_file=structure_file, research_goal=request, runner=software,
            force_field_files=force_field_files,
        )
        # Normalize the MD agent's single script_path into the common
        # input_files map so the pipeline stays engine-neutral downstream,
        # and record the entry script so the refinement loop knows what to run.
        script_path = result.get("script_path")
        if "input_files" not in result and script_path and Path(script_path).exists():
            result["input_files"] = {
                Path(script_path).name: Path(script_path).read_text()
            }
        if script_path:
            result["entry_file"] = Path(script_path).name
        result.setdefault("status", "success")
        return result

    raise ValueError(
        f"Unsupported simulation scale: {scale!r}. "
        f"Supported: {sorted(_DEFAULT_ENGINE)}. Adding a scale means a new "
        "foundation agent + skill bundle and one branch here."
    )


def _load_components_manifest(structure_path: str) -> Optional[Dict[str, Any]]:
    """Load a ``components.json`` manifest sitting next to a generated structure.

    Condensed structure generation writes this alongside the coordinate file:
    ``{"components": [{"name", "smiles", "count"}, ...]}`` in coordinate order.
    It is the force-field step's bridge from a packed box to per-species
    chemistry. Returns the manifest dict, or None when absent / unreadable (a
    crystal/molecular structure, an MLIP-MD run, or a caller-supplied data file
    has none — the FF step is then skipped).
    """
    if not structure_path:
        return None
    manifest = os.path.join(os.path.dirname(os.path.abspath(structure_path)),
                            "components.json")
    if not os.path.isfile(manifest):
        return None
    try:
        with open(manifest) as fh:
            data = json.load(fh)
        return data if data.get("components") else None
    except Exception:
        return None


def _parameterize_structure(
    structure_path: str,
    software: str,
    output_dir: str,
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
) -> Dict[str, Any]:
    """Turn a packed box into an engine-native, force-field-typed input.

    Runs the engine-neutral FF stack on the components.json manifest sitting
    next to ``structure_path``: ``ForceFieldAgent.parameterize`` ->
    ``ParameterizedSystem`` -> ``write_md_inputs``. Charges come from the
    manifest's SMILES (NAGL), so calling this once per member of a composition
    series with identical force-field arguments gives every member the same
    force field by construction.

    Returns a status dict:
      * ``{"status": "success", "structure_file", "force_field_files", "summary"}``
      * ``{"status": "skipped"}`` — no manifest (MLIP-MD / pre-built data file).
      * ``{"status": "error", "message"}`` — parameterization failed.
    """
    manifest = _load_components_manifest(structure_path)
    if not manifest:
        return {"status": "skipped"}
    try:
        from .force_field_agent import ForceFieldAgent
        from ._engine_inputs import write_md_inputs
        ff_agent = ForceFieldAgent(
            working_dir=output_dir, api_key=api_key,
            base_url=base_url, model_name=model_name,
        )
        psystem = ff_agent.parameterize(
            components=manifest["components"],
            coordinates_file=structure_path,
            working_dir=output_dir,
        )
        written = write_md_inputs(psystem, software, output_dir)
        return {
            "status": "success",
            "structure_file": written["structure_file"],
            "force_field_files": written["force_field_files"] or None,
            "summary": {
                "status": "success", "backend": psystem.backend,
                "n_atoms": psystem.n_atoms, "total_charge": psystem.total_charge,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_complete_workflow(
    user_request: str,
    *,
    scale: str = "periodic_dft",
    software: Optional[str] = None,
    method: str = "llm",
    structure_class: str = "crystal",
    output_dir: str = "simulation_workflow_output",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "claude-opus-4-6",
    futurehouse_api_key: Optional[str] = None,
    mp_api_key: Optional[str] = None,
    max_refinement_cycles: int = 4,
    script_timeout: int = 300,
    validate: bool = True,
    executor: "Executor | None" = None,
    run_command: Optional[str] = None,
    autonomy: str = "autonomous",
    max_run_cycles: int = 3,
    structure_file: Optional[str] = None,
    force_field_files: Optional[Dict[str, str]] = None,
    staged: bool = False,
) -> Dict[str, Any]:
    """Run the full structure → inputs → validation pipeline for any scale.

    Args:
        user_request: Natural-language description of the calculation.
        scale: Simulation scale (``"periodic_dft"``, ``"molecular_dynamics"``,
            …). Selects the foundation agent.
        software: Engine within the scale (e.g. ``"vasp"``, ``"lammps"``).
            Defaults to the scale's conventional engine.
        method: Input-generation backend. ``"llm"`` (default) uses the
            foundation agent's generation; a named backend (e.g.
            ``"atomate2"``) resolves to a skill-bundle generation tool.
        structure_class: Structure-class hint forwarded to structure
            generation.
        output_dir: Directory for all generated files.
        api_key, base_url, model_name: LLM credentials.
        futurehouse_api_key: Optional FutureHouse key enabling
            literature-grounded validation.
        mp_api_key: Optional Materials Project key for structure lookups.
        max_refinement_cycles: Structure validator-guided refinement cap.
        script_timeout: Timeout for executing generated structure scripts.
        validate: When True, run the pre-run InputValidator on the generated
            inputs (skipped for non-LLM methods, which are expert-defined).
        executor: Optional execution backend. When provided, the workflow runs
            the generated inputs and refines them to convergence via the
            engine-neutral refinement loop. ``LocalExecutor`` runs a local
            subprocess; ``ClusterExecutor`` (or ``ClusterExecutor.connect(...)``)
            submits to an HPC scheduler — the loop drives either through the same
            ``Executor`` contract. When ``None`` (the default, used for DFT), the
            workflow stops after generation + validation and the user runs the
            calculation externally.
        run_command: Command template the executor runs, with ``{script}``
            filled from each phase's entry file (e.g. ``"lmp -in {script}"``).
            User/config — required when ``executor`` is provided. The engine
            binary lives here, never in this module.
        autonomy: Autonomy level for the refinement loop (``"co-pilot"`` /
            ``"autopilot"`` / ``"autonomous"``); selects the built-in policy.
        max_run_cycles: Maximum run → assess → fix cycles per phase.
        structure_file: Optional path to an already-built structure. When
            provided, structure generation is skipped and this file is used
            directly — for callers that already have a structure and only want
            input generation + (optional) execution.
        force_field_files: Optional mapping of force-field filename to contents,
            forwarded to MD input generation.
        staged: When True, MD generation emits a multi-phase (optimization →
            equilibration → production) sequential campaign instead of a single
            run, so the refinement loop runs the per-phase loop over a restart-
            chained sequence. MD only; ignored by other scales.

    Returns:
        A workflow-result dict with ``final_status``, ``scale``, ``engine``,
        ``steps_completed``, ``output_directory``, and the per-step results
        (``structure_generation``, ``input_generation``, ``input_validation``).
    """
    software = software or _DEFAULT_ENGINE.get(scale)
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, Any] = {
        "user_request": user_request,
        "scale": scale,
        "engine": software,
        "steps_completed": [],
        "final_status": "started",
        "output_directory": output_dir,
    }

    # ── Step 1: structure generation + validation (scale-agnostic) ──
    # Skipped when the caller supplies an already-built structure.
    if structure_file is not None:
        structure_path = structure_file
        result["structure_generation"] = {
            "status": "skipped",
            "message": "caller-supplied structure",
            "final_structure_path": structure_file,
        }
    else:
        from .structure_pipeline import StructurePipeline
        structure = StructurePipeline(
            api_key=api_key, base_url=base_url, mp_api_key=mp_api_key,
            generator_model=model_name, validator_model=model_name,
            output_dir=output_dir, max_refinement_cycles=max_refinement_cycles,
            script_timeout=script_timeout,
        )
        # Reuse the structure pipeline's resolved credentials downstream.
        api_key = structure.api_key
        base_url = structure.base_url

        structure_result = structure.generate_and_validate(
            user_request, structure_class=structure_class,
        )
        result["structure_generation"] = structure_result
        if structure_result.get("status") != "success":
            result["final_status"] = "failed_structure_generation"
            return result
        result["steps_completed"].append("structure_generation")
        structure_path = structure_result["final_structure_path"]

    # ── Step 1.5: force-field parameterization (MD, force-field-based only) ──
    # Turn a packed box of coordinates into an engine-native, parameterized
    # input (e.g. a typed LAMMPS data file). Gated on a components.json manifest,
    # so MLIP-driven MD (potential-based, no manifest), pre-built data files, and
    # non-MD scales are untouched. When the caller already supplied
    # force_field_files, respect them.
    if (scale == "molecular_dynamics" and force_field_files is None):
        ff = _parameterize_structure(
            structure_path, software, output_dir,
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        if ff["status"] == "success":
            structure_path = ff["structure_file"]
            force_field_files = ff["force_field_files"] or None
            result["force_field"] = ff["summary"]
            result["steps_completed"].append("force_field")
        elif ff["status"] == "error":
            result["final_status"] = "failed_force_field"
            result["force_field"] = {"status": "error", "message": ff["message"]}
            return result
        else:  # "skipped" — no manifest
            result.setdefault("warnings", []).append(
                "molecular_dynamics run with no components.json manifest next to "
                "the structure and no force_field_files supplied — skipping "
                "force-field parameterization; the deck may read raw coordinates "
                "and fail to run."
            )

    # ── Step 2: input generation (routed to the scale's foundation agent) ──
    try:
        gen_result = _generate_inputs(
            scale=scale, software=software, method=method,
            structure_file=structure_path, request=user_request,
            output_dir=output_dir, api_key=api_key, base_url=base_url,
            model_name=model_name, force_field_files=force_field_files,
            staged=staged,
        )
    except Exception as e:
        result["final_status"] = "failed_input_generation"
        result["input_generation"] = {"status": "error", "message": str(e)}
        return result
    result["input_generation"] = gen_result
    if gen_result.get("status") not in (None, "success"):
        result["final_status"] = "failed_input_generation"
        return result
    result["steps_completed"].append("input_generation")

    # ── Step 3: pre-run input validation (engine-neutral critic) ──
    # Skipped for named (deterministic, expert-defined) backends and when
    # the caller opts out.
    if validate and method == "llm":
        input_files = _collect_input_files(gen_result)
        if input_files:
            from .critics import InputValidator
            validator = InputValidator(
                api_key=api_key, base_url=base_url, model_name=model_name,
                futurehouse_api_key=futurehouse_api_key,
            )
            result["input_validation"] = validator.validate(
                input_files=input_files, system_description=user_request,
                skill=software, domain=scale,
            )
            result["steps_completed"].append("input_validation")
    else:
        reason = ("non-LLM method uses expert-defined inputs"
                  if method != "llm" else "validation disabled by caller")
        result["input_validation"] = {"status": "skipped", "message": reason}

    # ── Step 4: supervised execution + refinement (only when an executor is
    # supplied; DFT's default executor=None stops here and runs externally) ──
    if executor is None:
        result["final_status"] = "success"
        return result

    if not run_command:
        result["refinement"] = {
            "status": "skipped",
            "message": "executor provided without a run_command template",
        }
        result["final_status"] = "success"
        return result

    from .refinement import RefinementContext, policy_for, run_campaign
    from .critics import RunCritic

    stages = _collect_stages(gen_result, output_dir, run_command)
    ctx = RefinementContext(
        research_goal=user_request, scale=scale, engine=software,
        skill=software, domain=scale, autonomy=autonomy,
        max_cycles=max_run_cycles,
    )
    run_critic = RunCritic(
        api_key=api_key, base_url=base_url, model_name=model_name,
    )
    refinement = run_campaign(
        stages, executor, run_critic, policy_for(autonomy), ctx,
        pre_run_verdict=result.get("input_validation"),
    )
    result["refinement"] = refinement
    result["steps_completed"].append("refinement")
    result["final_status"] = (
        "success" if refinement.get("status") == "success"
        else f"refinement_{refinement.get('status', 'failed')}"
    )
    return result


def _build_series_member(
    member: Dict[str, Any],
    *,
    software: str,
    density: Optional[float],
    member_dir: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
) -> Dict[str, Any]:
    """Build one composition-series member's force-field-typed structure.

    Two ways in, both landing on a typed engine input in ``member_dir``:
      * ``{"components": [...]}`` (+ optional ``density``/``box``) — packs a box
        with ``build_box`` (deterministic, seeded) then parameterizes it.
      * ``{"structure_file": path}`` — an already-typed structure, used as is.

    Returns ``{"status": "success", "structure_file", "force_field_files",
    "name", ...}`` or ``{"status": "error", "name", "message"}``.
    """
    name = member["name"]
    os.makedirs(member_dir, exist_ok=True)

    if member.get("structure_file"):
        # Pre-built, already-typed structure — trust it as the member's input.
        return {
            "status": "success", "name": name,
            "structure_file": member["structure_file"],
            "force_field_files": member.get("force_field_files"),
        }

    components = member.get("components")
    if not components:
        return {"status": "error", "name": name,
                "message": "member needs either 'components' or 'structure_file'"}

    try:
        from ...skills.structure_generation.condensed.build_box import build_box
        packed = build_box(
            components,
            density=member.get("density", density),
            box=member.get("box"),
            working_dir=member_dir,
        )
    except Exception as e:
        return {"status": "error", "name": name,
                "message": f"build_box failed: {e}"}

    # build_box writes components.json next to the coordinates, so the shared
    # force-field step turns the box into a typed engine input — identical FF
    # arguments across members means an identical force field by construction.
    ff = _parameterize_structure(
        packed["structure_file"], software, member_dir,
        api_key=api_key, base_url=base_url, model_name=model_name,
    )
    if ff["status"] == "error":
        return {"status": "error", "name": name,
                "message": f"parameterization failed: {ff['message']}"}
    if ff["status"] == "skipped":
        return {"status": "error", "name": name,
                "message": "build_box wrote no components.json manifest"}

    return {
        "status": "success", "name": name,
        "structure_file": ff["structure_file"],
        "force_field_files": ff["force_field_files"],
        "box": packed["box"], "density": packed["density"],
        "n_atoms": packed["n_atoms"], "n_molecules": packed["n_molecules"],
    }


def run_composition_series(
    user_request: str,
    members: List[Dict[str, Any]],
    *,
    scale: str = "molecular_dynamics",
    software: Optional[str] = None,
    density: Optional[float] = None,
    deck_from: int = 0,
    output_dir: str = "composition_series_output",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "claude-opus-4-6",
    futurehouse_api_key: Optional[str] = None,
    validate: bool = True,
    executor: "Executor | None" = None,
    run_command: Optional[str] = None,
    autonomy: str = "autonomous",
    max_run_cycles: int = 3,
) -> Dict[str, Any]:
    """Run one MD protocol across a series of compositions.

    A composition (or concentration) series: N members that share one
    force field and one protocol but differ in structure — the mirror of a
    temperature sweep, where members share a structure and differ in a deck
    scalar. Each member's box is packed and typed independently (in its own
    directory), then a single protocol deck is generated once and fanned out
    over every member, so the members differ *only* by composition and the
    comparison across them is controlled.

    The deck is generated once (from member ``deck_from``) and reused, which is
    what makes it one protocol rather than N independently-authored ones. This
    assumes the members share a species set — the deck's force-field styles must
    cover every member — which holds when composition varies by molecule *count*
    (the series case). A member that adds or drops a species needs its own deck
    and is a separate run, not a fan-out member.

    Args:
        user_request: Natural-language description of the protocol/goal.
        members: ``[{"name", "components"|"structure_file", "density"?, "box"?}]``.
            ``components`` is ``[{name, smiles, count}]`` packed by ``build_box``.
        scale: Simulation scale (``"molecular_dynamics"``).
        software: Engine within the scale; defaults to the scale's engine.
        density: Default target packing density (g/cm^3) for members that do not
            set their own. Only a starting configuration — the MD relaxes it.
        deck_from: Index of the member the shared protocol deck is generated
            from. Pick one whose species set covers the series.
        output_dir: Directory for per-member subdirectories and the campaign.
        api_key, base_url, model_name: LLM credentials.
        executor, run_command, autonomy, max_run_cycles: as in
            :func:`run_complete_workflow` — when ``executor`` is given the fan-out
            runs and refines, otherwise generation stops after the deck.

    Returns:
        A result dict with ``final_status``, ``members`` (per-member build
        results), and — when executed — ``refinement`` from the fan-out campaign.
    """
    software = software or _DEFAULT_ENGINE.get(scale)
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, Any] = {
        "user_request": user_request, "scale": scale, "engine": software,
        "mode": "composition_series", "output_directory": output_dir,
        "final_status": "started", "members": [],
    }

    if len(members) < 2:
        result["final_status"] = "failed_series"
        result["error"] = "a composition series needs at least two members"
        return result
    if not 0 <= deck_from < len(members):
        result["final_status"] = "failed_series"
        result["error"] = f"deck_from={deck_from} out of range for {len(members)} members"
        return result

    # ── Step 1+1.5: build + type every member independently ──
    built = []
    for member in members:
        member_dir = os.path.join(output_dir, _slugify(member["name"]))
        b = _build_series_member(
            member, software=software, density=density, member_dir=member_dir,
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        result["members"].append(b)
        if b["status"] != "success":
            result["final_status"] = "failed_member_build"
            result["error"] = f"member {b['name']!r}: {b.get('message')}"
            return result
        built.append(b)

    # ── Step 2: generate the shared protocol deck ONCE, from one member ──
    rep = built[deck_from]
    rep_dir = os.path.join(output_dir, _slugify(rep["name"]))
    try:
        gen = _generate_inputs(
            scale=scale, software=software, method="llm",
            structure_file=rep["structure_file"], request=user_request,
            output_dir=rep_dir, api_key=api_key, base_url=base_url,
            model_name=model_name, force_field_files=rep["force_field_files"],
        )
    except Exception as e:
        result["final_status"] = "failed_input_generation"
        result["error"] = f"deck generation failed: {e}"
        return result
    if gen.get("status") not in (None, "success"):
        result["final_status"] = "failed_input_generation"
        result["input_generation"] = gen
        return result

    entry = gen.get("entry_file") or "run.lammps"
    gen_files = gen.get("input_files") or {}
    deck_script = gen_files.get(entry, "")
    data_name = os.path.basename(rep["structure_file"])
    # Everything the deck needs except the structure is shared (identical force
    # field for all members); the structure is what each member overrides.
    shared_files = {
        name: contents for name, contents in gen_files.items()
        if name not in (entry, data_name)
    }

    # ── Step 3: fan the one deck out over the members' own structures ──
    fanout_members = []
    for b in built:
        with open(b["structure_file"]) as fh:
            typed = fh.read()
        fanout_members.append({
            "name": _slugify(b["name"]),
            "script": deck_script,
            # each member reads its OWN typed box under the deck's read_data name
            "files": {data_name: typed},
        })
    from .md_simulation_agent import _assemble_fanout_stage
    stages_spec = _assemble_fanout_stage(fanout_members, entry, shared_files)
    result["steps_completed"] = ["member_builds", "deck_generation"]
    result["deck_from"] = rep["name"]

    # ── Step 3.5: validate the shared protocol deck once (engine-neutral
    #    pre-run critic). One validation covers the protocol because the deck is
    #    shared across members; per-member structure validity is separately
    #    enforced at build time by the force-field completeness gate. The verdict
    #    feeds the refinement gate below, mirroring run_complete_workflow. ──
    if validate:
        deck_files = _collect_input_files(gen)
        if deck_files:
            from .critics import InputValidator
            validator = InputValidator(
                api_key=api_key, base_url=base_url, model_name=model_name,
                futurehouse_api_key=futurehouse_api_key,
            )
            result["input_validation"] = validator.validate(
                input_files=deck_files, system_description=user_request,
                skill=software, domain=scale,
            )
            result["steps_completed"].append("deck_validation")

    if executor is None:
        result["final_status"] = "generated"
        result["stages"] = stages_spec
        return result
    if not run_command:
        result["final_status"] = "generated"
        result["stages"] = stages_spec
        result["warnings"] = ["executor provided without a run_command; not run"]
        return result

    # ── Step 4: run + refine every member independently (fan-out campaign) ──
    from .refinement import RefinementContext, policy_for, run_campaign
    from .critics import RunCritic

    stages = _collect_stages({"stages": stages_spec}, output_dir, run_command)
    ctx = RefinementContext(
        research_goal=user_request, scale=scale, engine=software,
        skill=software, domain=scale, autonomy=autonomy,
        max_cycles=max_run_cycles,
    )
    refinement = run_campaign(
        stages, executor, RunCritic(api_key=api_key, base_url=base_url,
                                    model_name=model_name),
        policy_for(autonomy), ctx,
        pre_run_verdict=result.get("input_validation"),
    )
    result["refinement"] = refinement
    result["steps_completed"].append("refinement")
    result["final_status"] = (
        "success" if refinement.get("status") == "success"
        else f"refinement_{refinement.get('status', 'failed')}"
    )
    return result


def _collect_phases(
    gen_result: Dict[str, Any], run_dir: str, run_command_template: str
) -> list:
    """Build refinement ``Phase`` objects from a generation result.

    Reads only the normalized phase fields a foundation agent emits
    (``phases``, or an ``entry_file`` + ``input_files`` for single-phase
    engines), so no engine-specific keys appear here. The run command is the
    caller-provided template with ``{script}`` filled from each phase's entry
    file, so the engine binary is never assembled in this module.

    Args:
        gen_result: The input-generation result.
        run_dir: Directory the phases execute in (shared across phases so
            staged runs can read each other's restart files).
        run_command_template: Command template with an optional ``{script}``
            placeholder for the per-phase entry file.

    Returns:
        A list of ``Phase`` objects in execution order.
    """
    from .refinement import Phase

    phases_spec = gen_result.get("phases")
    if not phases_spec:
        entry = gen_result.get("entry_file")
        input_files = gen_result.get("input_files") or {}
        if entry is None and len(input_files) == 1:
            entry = next(iter(input_files))
        phases_spec = [{
            "name": "production",
            "input_files": input_files,
            "entry_file": entry,
        }]

    phases = []
    for spec in phases_spec:
        entry = spec.get("entry_file") or ""
        cmd = (
            run_command_template.format(script=entry)
            if "{script}" in run_command_template
            else run_command_template
        )
        phases.append(Phase(
            name=spec.get("name", "run"),
            input_files=spec.get("input_files") or {},
            run_command=cmd,
            run_dir=str(run_dir),
            entry_file=entry or "",
        ))
    return phases


def _collect_stages(
    gen_result: Dict[str, Any], run_dir: str, run_command_template: str
) -> list:
    """Build refinement ``Stage`` objects from a generation result.

    Reads only normalized, engine-neutral campaign fields. A generation result
    may carry a ``stages`` list describing a staged/parallel campaign; each
    entry is one of:

    * a **sequential step** — ``{name, input_files, entry_file}``. Steps share
      ``run_dir`` so restart files chain.
    * a **parallel fan-out** — ``{name, parallel: true, members: [...],
      min_success?}`` where each member is ``{name, input_files, entry_file}``.
      Members run in their own ``run_dir/<stage>/<member>`` directory.
    * a **combine** step — ``{name, kind: "combine", input_files, entry_file,
      run_command?}`` in ``run_dir/<stage>``; ``run_command`` may override the
      template (e.g. a Python post-processing script).

    When no ``stages`` field is present, the legacy single-chain shape is read
    via :func:`_collect_phases` and wrapped as one sequential stage, so older
    generation results behave exactly as before.

    Args:
        gen_result: The input-generation result.
        run_dir: Base directory the campaign executes in.
        run_command_template: Command template with an optional ``{script}``
            placeholder for a phase's entry file.

    Returns:
        A list of ``Stage`` objects in execution order.
    """
    import os

    from .refinement import Phase, Stage

    stages_spec = gen_result.get("stages")
    if not stages_spec:
        phases = _collect_phases(gen_result, run_dir, run_command_template)
        return [Stage(name="run", phases=phases, parallel=False)]

    def _command(entry: str, override) -> str:
        template = override or run_command_template
        if "{script}" in template:
            return template.format(script=entry or "")
        return template

    def _phase(spec: Dict[str, Any], rdir: str) -> "Phase":
        entry = spec.get("entry_file") or ""
        return Phase(
            name=spec.get("name", "run"),
            input_files=spec.get("input_files") or {},
            run_command=_command(entry, spec.get("run_command")),
            run_dir=str(rdir),
            entry_file=entry or "",
        )

    stages = []
    for spec in stages_spec:
        name = spec.get("name", "run")
        if spec.get("kind") == "combine":
            stages.append(Stage(
                name=name, kind="combine", parallel=False,
                phases=[_phase(spec, os.path.join(str(run_dir), name))],
            ))
        elif spec.get("parallel") or spec.get("members"):
            members = [
                _phase(m, os.path.join(str(run_dir), name,
                                       m.get("name", "member")))
                for m in (spec.get("members") or [])
            ]
            stages.append(Stage(
                name=name, parallel=True, phases=members,
                min_success=spec.get("min_success"),
            ))
        else:
            # Sequential step: share the base run_dir so restart files chain.
            stages.append(Stage(
                name=name, parallel=False, phases=[_phase(spec, run_dir)],
            ))
    return stages


def _collect_input_files(gen_result: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{filename: contents}`` from a generation result.

    Reads the normalized ``input_files`` map every ``_generate_inputs``
    branch produces. Values may be inlined contents or paths; paths are
    read so the InputValidator always receives contents. No engine-specific
    filenames are assumed.
    """
    contents: Dict[str, str] = {}
    files = gen_result.get("input_files")
    if not isinstance(files, dict):
        return contents
    for name, val in files.items():
        if not isinstance(val, str):
            continue
        try:
            p = Path(val)
            if p.exists():
                contents[name] = p.read_text()
                continue
        except (OSError, ValueError):
            pass
        contents[name] = val
    return contents
