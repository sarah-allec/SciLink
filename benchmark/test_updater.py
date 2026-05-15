"""VaspUpdater test — fold examples/breakage_benchmark_* into the suite.

For each planted-error case in ``breakage_manifest.json``, feed the
INCAR + POSCAR + KPOINTS + vasp log to ``VaspUpdater.refine_inputs``
and score:

  * Did refine_inputs return ``status == "success"``?
  * Did the proposed fix touch every key in ``expected_fix_keys`` from
    the breakage manifest? (e.g. ZBRENT case → IBRION + POTIM)
  * Was the fix deterministic (no LLM call) or LLM-driven?

Pass criterion per case:  status success **and** every expected key
present in the fix.

Local; cost is one updater call per case.  Two layers: deterministic
patterns first, LLM fallback for anything not in the catalog — both
exercise different code paths and both are valid pass routes.

Examples:
    python -m benchmark.runner test_updater --dry-run
    python -m benchmark.runner test_updater --case zbrent
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._score import CaseResult, RunnerManifest, write_manifest, write_summary_md


_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_BENCH = (_REPO / "examples" / "breakage_benchmark_20260510_075135")


# ──────────────────────────────────────────────────────────────────
#  Fixture loader  —  reads breakage_manifest.json
# ──────────────────────────────────────────────────────────────────

def _load_breakage_cases(root: Path) -> List[Dict[str, Any]]:
    """Return the case list from the breakage manifest, with absolute
    case_dir paths resolved against ``root``."""
    mpath = root / "breakage_manifest.json"
    if not mpath.is_file():
        return []
    with open(mpath) as f:
        data = json.load(f)
    cases = []
    for c in data.get("cases", []):
        case_dir = c.get("case_dir", "")
        # The manifest stores absolute paths from the day it was generated;
        # repair them against the current repo location if they don't exist.
        if not os.path.isdir(case_dir):
            case_dir = str(root / c["label"])
        cases.append({
            "label": c["label"],
            "case_dir": case_dir,
            "targets": c.get("targets", ""),
            "expected_fix_keys": c.get("expected_fix_keys", []),
            "overrides": c.get("overrides", {}),
        })
    return cases


# ──────────────────────────────────────────────────────────────────
#  Scoring  —  did the updater hit the expected fix keys?
# ──────────────────────────────────────────────────────────────────

def _extract_fixed_keys(refine_result: Dict[str, Any]) -> List[str]:
    """Pull the set of INCAR keys the updater claims to have touched."""
    expl = refine_result.get("explanation") or {}
    fixes = expl.get("fixes_applied")
    keys: List[str] = []
    if isinstance(fixes, dict):
        keys.extend(fixes.keys())
    elif isinstance(fixes, list):
        for item in fixes:
            if isinstance(item, dict) and "key" in item:
                keys.append(item["key"])
            elif isinstance(item, str):
                keys.append(item.split("=")[0].strip())
    # If suggested_incar exists but no structured fixes_applied is given,
    # fall back to diffing the suggested INCAR against the original later.
    return [k.upper() for k in keys]


def _diff_incar_keys(original_text: str, suggested_text: str) -> List[str]:
    """Keys whose value changed (or that were added) between original and
    suggested INCAR."""
    def _parse(t):
        out = {}
        for line in t.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip().upper()] = v.strip()
        return out
    o = _parse(original_text)
    s = _parse(suggested_text)
    return sorted(
        {k for k in s if s[k] != o.get(k)} | {k for k in o if k not in s}
    )


def _score_one(case: Dict[str, Any],
               refine_result: Dict[str, Any],
               original_incar: str) -> CaseResult:
    expected_keys = [k.upper() for k in case["expected_fix_keys"]]
    status = refine_result.get("status")
    method = refine_result.get("method", "unknown")

    # Set of keys the updater claims to have changed.
    fixed = set(_extract_fixed_keys(refine_result))
    if not fixed and "suggested_incar" in refine_result:
        fixed = set(_diff_incar_keys(original_incar,
                                     refine_result["suggested_incar"]))

    expected_hit = set(expected_keys) & fixed
    expected_missed = set(expected_keys) - fixed

    success = (status == "success") and not expected_missed
    score = (
        1.0 if success
        else (0.5 if status == "success" and expected_hit
              else 0.0)
    )
    return CaseResult(
        id=case["label"],
        expected={"targets": case["targets"],
                  "expected_fix_keys": expected_keys},
        actual={
            "status": status,
            "method": method,
            "keys_fixed": sorted(fixed),
            "expected_hit": sorted(expected_hit),
            "expected_missed": sorted(expected_missed),
            "explanation": refine_result.get("explanation", {}),
        },
        score=score,
        passed=success,
        notes=(
            f"method={method}; hit={len(expected_hit)}/{len(expected_keys)} "
            f"expected keys"
        ),
    )


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def _load_case_inputs(case_dir: Path) -> Optional[Dict[str, str]]:
    """Read INCAR, POSCAR, KPOINTS, and a VASP log from the case dir.
    Returns None if anything required is missing."""
    needed = {
        "poscar": case_dir / "POSCAR",
        "incar":  case_dir / "INCAR",
        "kpoints": case_dir / "KPOINTS",
    }
    for k, p in needed.items():
        if not p.is_file():
            return None
    # vasp log:  prefer vasp.out;  fall back to slurm-*.out;  empty string
    # if neither exists (the updater can still run on INCAR alone for some
    # error classes, but score may suffer).
    log_text = ""
    for cand in ("vasp.out", "stdout", "vasp.log"):
        p = case_dir / cand
        if p.is_file():
            log_text = p.read_text(errors="replace")
            break
    if not log_text:
        for p in case_dir.glob("slurm-*.out"):
            log_text = p.read_text(errors="replace")
            break
    return {
        "poscar_path":  str(needed["poscar"]),
        "incar_path":   str(needed["incar"]),
        "kpoints_path": str(needed["kpoints"]),
        "vasp_log":     log_text,
        "incar_text":   needed["incar"].read_text(),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_updater", description=__doc__)
    parser.add_argument("--model",    default="claude-sonnet-4-5")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--breakage-root", default=str(_DEFAULT_BENCH),
                        help="path to examples/breakage_benchmark_<ts>")
    parser.add_argument("--case",     default=None,
                        help="run only this case (label)")
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--out-dir",  default="benchmark/outputs/test_updater")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.breakage_root)
    cases = _load_breakage_cases(root)
    if args.case:
        cases = [c for c in cases if c["label"] == args.case]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print(f"!! no cases found under {root}", file=sys.stderr)
        return 2

    print(f"test_updater :: {len(cases)} cases  root={root}  model={args.model}")
    if args.dry_run:
        for c in cases:
            inputs = _load_case_inputs(Path(c["case_dir"]))
            tag = "ready" if inputs else "MISSING inputs"
            print(f"  [{tag:<14}] {c['label']:<14}  targets={c['targets']}  "
                  f"expect_keys={c['expected_fix_keys']}")
        return 0

    api_key = args.api_key or os.environ.get("SCILINK_API_KEY")
    base_url = args.base_url or os.environ.get("SCILINK_BASE_URL")

    from scilink.agents.sim_agents.vasp_updater import VaspUpdater
    updater = VaspUpdater(api_key=api_key, base_url=base_url,
                          model_name=args.model)

    manifest = RunnerManifest(runner="test_updater", mode="local")

    for c in cases:
        case_dir = Path(c["case_dir"])
        inputs = _load_case_inputs(case_dir)
        if inputs is None:
            print(f"  !! {c['label']}: missing required inputs in {case_dir}")
            manifest.append(CaseResult(
                id=c["label"],
                expected={"expected_fix_keys": c["expected_fix_keys"]},
                actual={"error": f"missing inputs in {case_dir}"},
                score=0.0, passed=False,
                notes="case dir missing one of POSCAR/INCAR/KPOINTS",
            ))
            continue

        try:
            res = updater.refine_inputs(
                poscar_path=inputs["poscar_path"],
                incar_path=inputs["incar_path"],
                kpoints_path=inputs["kpoints_path"],
                vasp_log=inputs["vasp_log"],
                original_request=(
                    "Cubic-bulk relaxation that errored out; recover "
                    "by adjusting the appropriate INCAR keys."
                ),
            )
        except Exception as exc:
            res = {"status": "error", "explanation": {"exception": repr(exc)}}

        case = _score_one(c, res, inputs["incar_text"])

        # Stash the suggested INCAR for inspection.
        out_case = Path(args.out_dir) / c["label"]
        out_case.mkdir(parents=True, exist_ok=True)
        if "suggested_incar" in res:
            (out_case / "INCAR.suggested").write_text(res["suggested_incar"])

        manifest.append(case)
        flag = "✓" if case.passed else ("~" if case.score > 0 else "✗")
        print(f"  {flag} {c['label']:<14} method={case.actual.get('method'):<14} "
              f"hit={case.actual['expected_hit']}  miss={case.actual['expected_missed']}")

    if manifest.cases:
        det_n = sum(c.actual.get("method") == "deterministic"
                    for c in manifest.cases)
        manifest.metrics = {
            "pass_rate":             manifest.passed / manifest.n_cases,
            "deterministic_fraction": det_n / manifest.n_cases,
            "mean_score":            sum(c.score for c in manifest.cases)
                                     / manifest.n_cases,
        }
    write_manifest(args.out_dir, manifest)
    write_summary_md(args.out_dir, manifest)
    print(f"\nwrote {args.out_dir}/manifest.json (+summary.md)")
    if manifest.metrics:
        print(f"pass rate: {manifest.metrics['pass_rate']:.2f}  "
              f"deterministic: {manifest.metrics['deterministic_fraction']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
