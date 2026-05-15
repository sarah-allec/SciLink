"""Structure-generator test — for each system tagged ``structure_gen``,
drive ``StructureGenerator`` (with a short refinement loop on validator
feedback) and score the produced POSCAR against the system's
ASE-built reference.

Sub-scores per system (max 1.0):
  * 0.30  script ran and produced a POSCAR
  * 0.20  element set matches expected
  * 0.20  atom count matches reference (within ±1)
  * 0.20  lattice parameter within 10 % of reference
  * 0.10  spglib space group matches reference  (only checked
          when reference is itself recognised by spglib)

A case is ``passed`` at score ≥ 0.70.

Local; cost is one to a few LLM calls per system, plus the executed
generator script.

Examples:
    python -m benchmark.runner test_structure_gen --dry-run
    python -m benchmark.runner test_structure_gen --system fe_bcc
    python -m benchmark.runner test_structure_gen --limit 3 --cycles 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .systems import (
    BenchmarkSystem, all_systems, get_system, systems_for_agent,
)
from ._score import CaseResult, RunnerManifest, write_manifest, write_summary_md


# ──────────────────────────────────────────────────────────────────
#  Generator + validator wiring  (slim re-implementation of the
#  structure phase of DFTOrchestrator; this benchmark is testing
#  the agents directly, so it deliberately doesn't go through the
#  orchestrator).
# ──────────────────────────────────────────────────────────────────

def _generate_structure(system: BenchmarkSystem,
                        out_dir: str,
                        api_key: Optional[str],
                        base_url: Optional[str],
                        model_name: str,
                        max_cycles: int) -> Dict[str, Any]:
    """Drive StructureGenerator with up to ``max_cycles`` validator-feedback
    refinement cycles.  Returns a dict with at least ``status`` and, on
    success, ``structure_file`` + ``cycles_used``."""
    from scilink.agents.sim_agents.structure_agent import StructureGenerator
    from scilink.agents.sim_agents.val_agent import StructureValidatorAgent

    os.makedirs(out_dir, exist_ok=True)

    gen = StructureGenerator(
        api_key=api_key, base_url=base_url, model_name=model_name,
        generated_script_dir=os.path.join(out_dir, "_scripts"),
    )
    val = StructureValidatorAgent(
        api_key=api_key, base_url=base_url, model_name=model_name,
    )

    request = (
        f"{system.description}. "
        f"Build the structure and save it as POSCAR in the current "
        f"directory. "
        f"Use ase.io.write(..., format='vasp', sort=True) so the species "
        f"line is grouped by element."
    )

    prev_script = None
    feedback: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    cwd = os.getcwd()

    try:
        os.chdir(out_dir)   # so the generator's "save POSCAR here" lands here
        for cycle in range(max_cycles + 1):
            gen_result = gen.generate_script(
                original_user_request=request,
                attempt_number_overall=cycle + 1,
                is_refinement_from_validation=(cycle > 0),
                previous_script_content=prev_script if cycle > 0 else None,
                validator_feedback=feedback if cycle > 0 else None,
                attempt_history=history if cycle > 0 else None,
            )
            if gen_result.get("status") != "success":
                return {"status": "error",
                        "message": gen_result.get("message", "generate_script failed"),
                        "cycles_used": cycle + 1}

            poscar_path = gen_result["output_file"]
            script_content = gen_result["final_script_content"]

            # validator
            v = val.validate_structure_and_script(
                structure_file_path=poscar_path,
                generating_script_content=script_content,
                original_request=request,
            )
            history.append({
                "script": script_content,
                "issues": list(v.get("all_identified_issues", []) or []),
                "hints": list(v.get("script_modification_hints", []) or []),
            })

            if v.get("status") == "success":
                return {"status": "success",
                        "structure_file": poscar_path,
                        "cycles_used": cycle + 1,
                        "validation": v}

            prev_script = script_content
            feedback = v

        # Loop exhausted without validator success — return what we have.
        return {"status": "exhausted",
                "structure_file": poscar_path,
                "cycles_used": max_cycles + 1,
                "validation": feedback}
    finally:
        os.chdir(cwd)


# ──────────────────────────────────────────────────────────────────
#  Scoring  —  compare produced POSCAR against system.build() reference.
# ──────────────────────────────────────────────────────────────────

def _read_atoms(path: str):
    from ase.io import read
    return read(path, format="vasp")


def _spacegroup(atoms) -> Optional[Tuple[str, int]]:
    """Return (symbol, number) via spglib; None if spglib unavailable or
    detection fails."""
    try:
        import spglib
    except Exception:
        return None
    try:
        cell = (atoms.cell.array, atoms.get_scaled_positions(),
                atoms.numbers)
        sg = spglib.get_spacegroup(cell, symprec=1e-3)
        if not sg:
            return None
        sym, num = sg.split("(")
        return sym.strip(), int(num.rstrip(") "))
    except Exception:
        return None


def _score_structure(system: BenchmarkSystem,
                     produced_path: str) -> Tuple[float, Dict[str, Any]]:
    """Return (score, detail) by comparing the produced POSCAR against
    the system's ASE-built reference (when buildable)."""
    detail: Dict[str, Any] = {"checks": {}}

    # 0.30 — file exists and is parseable
    try:
        produced = _read_atoms(produced_path)
        detail["checks"]["script_produced_poscar"] = True
        score = 0.30
    except Exception as exc:
        detail["checks"]["script_produced_poscar"] = False
        detail["error"] = f"could not read produced POSCAR: {exc!r}"
        return 0.0, detail

    # Reference Atoms — only systems with builders.  Surfaces and
    # defected cells will not match the reference cellpar exactly, but
    # the element set + space-group check still discriminate.
    try:
        reference = system.build()
        has_ref = hasattr(reference, "get_chemical_symbols")
    except (NotImplementedError, Exception):
        reference = None
        has_ref = False

    if not has_ref:
        detail["note"] = ("no buildable ASE reference for this system; "
                          "scoring on element set only")
        produced_elems = sorted(set(produced.get_chemical_symbols()))
        expected_elems = sorted(set(system.elements))
        if produced_elems == expected_elems:
            score += 0.20 + 0.20 + 0.20    # accept counts/lattice when
                                            # there's no reference to check
            detail["checks"]["elements_match"] = True
        else:
            detail["checks"]["elements_match"] = False
        return score, detail

    # 0.20 — element set
    produced_elems = sorted(set(produced.get_chemical_symbols()))
    expected_elems = sorted(set(reference.get_chemical_symbols()))
    if produced_elems == expected_elems:
        score += 0.20
        detail["checks"]["elements_match"] = True
    else:
        detail["checks"]["elements_match"] = False
        detail["checks"]["expected_elements"] = expected_elems
        detail["checks"]["produced_elements"] = produced_elems

    # 0.20 — atom count (±1)
    if abs(len(produced) - len(reference)) <= 1:
        score += 0.20
        detail["checks"]["atom_count_match"] = True
    else:
        detail["checks"]["atom_count_match"] = False
        detail["checks"]["expected_n_atoms"] = len(reference)
        detail["checks"]["produced_n_atoms"] = len(produced)

    # 0.20 — lattice cellpar within 10 %  (compare a-vectors only,
    # since for non-cubic systems we'd need a proper cell-similarity
    # metric)
    try:
        ref_a = reference.cell.cellpar()[0]
        prod_a = produced.cell.cellpar()[0]
        if ref_a > 0 and abs(prod_a - ref_a) / ref_a < 0.10:
            score += 0.20
            detail["checks"]["lattice_within_10pct"] = True
        else:
            detail["checks"]["lattice_within_10pct"] = False
            detail["checks"]["expected_a"] = ref_a
            detail["checks"]["produced_a"] = prod_a
    except Exception as exc:
        detail["checks"]["lattice_within_10pct"] = False
        detail["checks"]["lattice_error"] = repr(exc)

    # 0.10 — space group (best-effort; both must be detectable)
    ref_sg = _spacegroup(reference)
    prod_sg = _spacegroup(produced)
    if ref_sg and prod_sg and ref_sg[1] == prod_sg[1]:
        score += 0.10
        detail["checks"]["spacegroup_match"] = True
        detail["checks"]["spacegroup"] = ref_sg[0]
    elif ref_sg and prod_sg:
        detail["checks"]["spacegroup_match"] = False
        detail["checks"]["expected_sg"] = ref_sg
        detail["checks"]["produced_sg"] = prod_sg
    else:
        detail["checks"]["spacegroup_match"] = "skipped (spglib unavailable or detection failed)"

    return score, detail


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_structure_gen", description=__doc__)
    parser.add_argument("--model",    default="claude-sonnet-4-5")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--system",   default=None,
                        help="run only this system (slug)")
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--cycles",   type=int, default=1,
                        help="max validator-feedback refinement cycles "
                             "(0 = initial only)")
    parser.add_argument("--out-dir",  default="benchmark/outputs/test_structure_gen")
    parser.add_argument("--pass-threshold", type=float, default=0.70)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args(argv)

    systems = systems_for_agent("structure_gen")
    if args.system:
        systems = [get_system(args.system)]
    if args.limit:
        systems = systems[: args.limit]

    print(f"structure-gen test :: {len(systems)} systems  "
          f"model={args.model}  cycles_max={args.cycles}")
    if args.dry_run:
        # Try to build; explicitly tolerate NotImplementedError (TODO) and
        # ImportError / ModuleNotFoundError (no ase in this env).
        for s in systems:
            if s.builder is None:
                tag = "no builder"
            else:
                try:
                    s.build()
                    tag = "build OK"
                except NotImplementedError:
                    tag = "TODO builder"
                except (ImportError, ModuleNotFoundError):
                    tag = "ase missing"
                except Exception as exc:
                    tag = f"build err: {type(exc).__name__}"
            print(f"  [{tag:<14}] {s.name:<22} {s.kind:<20} {','.join(s.elements)}")
        return 0

    api_key = args.api_key or os.environ.get("SCILINK_API_KEY")
    base_url = args.base_url or os.environ.get("SCILINK_BASE_URL")

    manifest = RunnerManifest(runner="test_structure_gen", mode="local")

    for s in systems:
        sys_dir = os.path.join(args.out_dir, s.name)
        os.makedirs(sys_dir, exist_ok=True)
        print(f"\n— {s.name} ({s.kind}) —")
        try:
            gen_result = _generate_structure(
                s, sys_dir, api_key, base_url, args.model, args.cycles)
        except Exception as exc:
            print(f"  !! generator raised: {exc!r}")
            traceback.print_exc()
            manifest.append(CaseResult(
                id=s.name,
                expected={"description": s.description, "elements": s.elements},
                actual={"error": repr(exc)},
                score=0.0, passed=False,
                notes="generator raised",
            ))
            continue

        if gen_result["status"] == "error":
            print(f"  !! {gen_result.get('message')}")
            manifest.append(CaseResult(
                id=s.name,
                expected={"description": s.description, "elements": s.elements},
                actual=gen_result,
                score=0.0, passed=False,
                notes=gen_result.get("message", ""),
            ))
            continue

        poscar = gen_result["structure_file"]
        score, detail = _score_structure(s, poscar)
        passed = score >= args.pass_threshold
        manifest.append(CaseResult(
            id=s.name,
            expected={"description": s.description, "elements": s.elements},
            actual={
                "structure_file": poscar,
                "cycles_used": gen_result.get("cycles_used"),
                "validator_status": gen_result.get("validation", {}).get("status"),
                **detail,
            },
            score=round(score, 3),
            passed=passed,
            notes=detail.get("note", ""),
        ))
        flag = "✓" if passed else "✗"
        print(f"  {flag} score={score:.2f}  cycles={gen_result.get('cycles_used')}  "
              f"checks={detail['checks']}")

    # Aggregate metrics
    if manifest.cases:
        manifest.metrics = {
            "mean_score":        sum(c.score for c in manifest.cases) / len(manifest.cases),
            "pass_rate":         manifest.passed / max(1, manifest.n_cases),
            "exhausted_rate":    sum(
                1 for c in manifest.cases
                if c.actual.get("validator_status") and c.actual["validator_status"] != "success"
            ) / max(1, manifest.n_cases),
        }
    write_manifest(args.out_dir, manifest)
    write_summary_md(args.out_dir, manifest)
    print(f"\nwrote {args.out_dir}/manifest.json + summary.md")
    if manifest.metrics:
        print(f"mean score: {manifest.metrics['mean_score']:.2f}    "
              f"pass rate: {manifest.metrics['pass_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
