"""VaspQualityAgent test — score the post-run quality assessor on a
curated set of known-good + known-bad VASP outputs.

Fixtures come from two places:
  * ``validation/results/<mode>/<system>/vasp/``   — 12 cells from the
    lattice-constant panel (9 good, 3 MgO POSCAR-bug failures).
  * ``examples/breakage_benchmark_20260510_075135/<class>/`` — planted
    error classes (low_nbands, low_nelm, zbrent).

Scoring is a 2 × 2 confusion matrix on the agent's verdict vs the label:
  * label ``good``     ↔ agent says ``success``      (TN if matched)
  * label ``critical`` ↔ agent says ``critical``    (TP if matched)
  * a ``warning`` label is treated as "must flag any issue at all"

Pass criterion per case: agent's status matches the expected category
(success ↔ good; any-issue ↔ warning; critical ↔ critical).

Examples:
    python -m benchmark.runner test_quality --dry-run
    python -m benchmark.runner test_quality --limit 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._score import CaseResult, RunnerManifest, write_manifest, write_summary_md


# ──────────────────────────────────────────────────────────────────
#  Fixture set
# ──────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent

_VAL = _REPO / "validation" / "results"
_BRK = _REPO / "examples" / "breakage_benchmark_20260510_075135"


# Each fixture: (id, path, expected, label, research_goal)
# expected: "good" | "warning" | "critical"
_BASELINE_GOAL = (
    "Relax the cell and ions of a cubic bulk crystal and report the "
    "equilibrium lattice constant."
)

FIXTURES: List[Dict[str, Any]] = [

    # ── validation panel runs (good cases) ────────────────────────
    *[{
        "id": f"val_{mode}_{sys}",
        "path": str(_VAL / mode / sys / "vasp"),
        "expected": "good",
        "label": f"{sys} under {mode} — converged, lattice within +0.7 % of exp",
        "research_goal": _BASELINE_GOAL,
    }
        for mode in ("a_forced", "b_agent_select", "c_bare_goal")
        for sys in ("si_diamond", "cu_fcc", "c_diamond")
    ],

    # ── validation panel runs (bad: MgO POSCAR-sort failure) ──────
    *[{
        "id": f"val_{mode}_mgo",
        "path": str(_VAL / mode / "mgo_rocksalt" / "vasp"),
        "expected": "critical",
        "label": f"mgo_rocksalt under {mode} — VASP refused (POSCAR/POTCAR mismatch)",
        "research_goal": _BASELINE_GOAL,
    }
        for mode in ("a_forced", "b_agent_select", "c_bare_goal")
    ],

    # ── planted breakage benchmark cases (post-failure / post-fix) ─
    {
        "id": "brk_low_nbands",
        "path": str(_BRK / "low_nbands"),
        "expected": "critical",
        "label": "planted: NBANDS set so low the calc cannot converge",
        "research_goal": _BASELINE_GOAL,
    },
    {
        "id": "brk_low_nelm",
        "path": str(_BRK / "low_nelm"),
        "expected": "warning",
        "label": "planted: NELM too small → SCF doesn't converge per step",
        "research_goal": _BASELINE_GOAL,
    },
    {
        "id": "brk_zbrent",
        "path": str(_BRK / "zbrent"),
        "expected": "critical",
        "label": "planted: ZBRENT trust-radius failure in ionic relaxation",
        "research_goal": _BASELINE_GOAL,
    },
]


# ──────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────

def _verdict_matches(expected: str, agent_status: str) -> bool:
    """Map agent status onto the label categories.

    Agent statuses we expect:
      * "success" / "ok"            → "good"
      * "warning"                   → "warning"
      * "critical" / "error" / etc. → "critical"
    """
    s = (agent_status or "").lower()
    if expected == "good":
        return s in ("success", "ok", "passed")
    if expected == "warning":
        return s in ("warning", "critical", "error")   # any flag counts
    if expected == "critical":
        return s in ("critical", "error", "fatal")
    return False


def _check_path_exists(fixture: Dict[str, Any]) -> bool:
    """Filter out fixtures whose paths aren't rsynced locally."""
    p = Path(fixture["path"])
    if not p.is_dir():
        return False
    # at minimum we need an OUTCAR (the analyzer reads it via pymatgen);
    # zero-byte OUTCARs (VASP refused at startup) are still valid inputs
    # — the agent should mark them critical.
    return (p / "OUTCAR").exists() or (p / "vasprun.xml").exists()


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_quality", description=__doc__)
    parser.add_argument("--model",    default="claude-sonnet-4-5")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--expected", default=None,
                        choices=["good", "warning", "critical"],
                        help="only score fixtures with this label")
    parser.add_argument("--out-dir",  default="benchmark/outputs/test_quality")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args(argv)

    fixtures = list(FIXTURES)
    if args.expected:
        fixtures = [f for f in fixtures if f["expected"] == args.expected]
    if args.limit:
        fixtures = fixtures[: args.limit]

    print(f"quality test :: {len(fixtures)} fixtures  model={args.model}")
    for f in fixtures:
        exists = _check_path_exists(f)
        tag = "OK" if exists else "MISSING"
        if args.dry_run:
            print(f"  [{tag:<8}] [{f['expected']:<8}] {f['id']:<24}  {f['path']}")
    if args.dry_run:
        return 0

    # Filter to only the fixtures that exist locally.
    runnable = [f for f in fixtures if _check_path_exists(f)]
    if not runnable:
        print("\nno fixtures available — did you rsync validation/results/ "
              "and examples/breakage_benchmark_*?", file=sys.stderr)
        return 2
    if len(runnable) < len(fixtures):
        print(f"\nskipping {len(fixtures) - len(runnable)} missing fixture(s)")

    # Build the quality agent.
    from scilink.agents.sim_agents.vasp_quality import VaspQualityAgent
    api_key = args.api_key or os.environ.get("SCILINK_API_KEY")
    base_url = args.base_url or os.environ.get("SCILINK_BASE_URL")
    agent = VaspQualityAgent(
        api_key=api_key, base_url=base_url, model_name=args.model,
    )

    manifest = RunnerManifest(runner="test_quality", mode="local")

    for f in runnable:
        try:
            result = agent.run_quality_check(
                output_dir=f["path"],
                research_goal=f["research_goal"],
            )
            actual_status = result.get("status", "unknown")
            issues = result.get("issues", []) or []
        except Exception as exc:
            actual_status = "error"
            issues = [{"description": f"agent raised: {exc!r}"}]
            result = {}

        matched = _verdict_matches(f["expected"], actual_status)
        case = CaseResult(
            id=f["id"],
            expected={"verdict": f["expected"], "label": f["label"]},
            actual={
                "status": actual_status,
                "n_issues": len(issues),
                "issues_summary": [
                    {"severity": i.get("severity"),
                     "category": i.get("category"),
                     "description": (i.get("description") or "")[:140]}
                    for i in issues[:5]   # cap for manifest size
                ],
            },
            score=1.0 if matched else 0.0,
            passed=matched,
        )
        manifest.append(case)
        flag = "✓" if matched else "✗"
        print(f"  {flag} [{f['expected']:<8}] {f['id']:<24}  "
              f"got={actual_status:<10}  n_issues={len(issues)}")

    # confusion-matrix style metrics
    by_label = {"good": [], "warning": [], "critical": []}
    for f, c in zip(runnable, manifest.cases):
        by_label.setdefault(f["expected"], []).append(c.passed)
    metrics: Dict[str, float] = {
        "overall_accuracy": manifest.passed / max(1, manifest.n_cases),
    }
    for lab, lst in by_label.items():
        if lst:
            metrics[f"accuracy_{lab}"] = sum(lst) / len(lst)
    manifest.metrics = metrics

    write_manifest(args.out_dir, manifest)
    write_summary_md(args.out_dir, manifest)
    print(f"\nwrote {args.out_dir}/manifest.json + summary.md")
    print(f"overall accuracy: {metrics['overall_accuracy']:.2f}")
    for lab in ("good", "warning", "critical"):
        if f"accuracy_{lab}" in metrics:
            print(f"  {lab:<8} accuracy: {metrics[f'accuracy_{lab}']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
