"""Router test — fires every ``RouterQuery`` at ``SimulationRouter`` and
scores top-1 accuracy on scale, engine, and joint (both correct).

Runs locally; the only cost is ~25 LLM calls.

Examples:
    python -m benchmark.runner test_router                 # full sweep
    python -m benchmark.runner test_router --limit 5       # quick probe
    python -m benchmark.runner test_router --difficulty hard
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from .queries import ALL_QUERIES, RouterQuery, by_difficulty, by_scale
from ._score import CaseResult, RunnerManifest, write_manifest, write_summary_md


# ──────────────────────────────────────────────────────────────────
#  Model wiring  —  same pattern the simulate orchestrator uses.
# ──────────────────────────────────────────────────────────────────

def _build_model(model_name: str,
                 api_key: Optional[str],
                 base_url: Optional[str]):
    """Same selection logic the simulation agents use:
      * ``base_url`` set        → OpenAI-compatible wrapper (PNNL proxy).
      * no ``base_url``         → LiteLLM, which routes Anthropic /
                                  Google / OpenAI by model-name inference.

    Doing it the other way around (OpenAI first regardless) ends up
    POSTing an Anthropic-shape key (``sk-ant-…``) at api.openai.com and
    failing with a 401.
    """
    api_key = api_key or os.environ.get("SCILINK_API_KEY")
    base_url = base_url or os.environ.get("SCILINK_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "no API key — set SCILINK_API_KEY or pass --api-key")
    if base_url:
        from scilink.wrappers.openai_wrapper import OpenAIAsGenerativeModel
        return OpenAIAsGenerativeModel(
            model=model_name, api_key=api_key, base_url=base_url)
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    return LiteLLMGenerativeModel(
        model=model_name, api_key=api_key, base_url=None)


def _build_mock_software():
    """Return an AvailableSoftware that marks every agent-supported engine
    as available.

    Why this exists:  the router's ``candidate_engines()`` intersects
    "what the agent supports" with "what the user has installed" — and on
    a laptop running this benchmark we generally don't have VASP / LAMMPS /
    MACE locally.  That makes the candidate set empty and the router
    returns ``(None, None)`` for every prompt.

    The benchmark measures the LLM's *decision quality* on the full
    possibility space, not whether your laptop has the engines.  So we
    override the candidate filter with "everything supported is
    available" and let the LLM pick freely.

    Pass ``--use-real-software`` to skip this and use ``AvailableSoftware.auto()``
    — useful when running on the cluster to score against the actual
    installed set.
    """
    from scilink.utils.available_software import AvailableSoftware
    from scilink.agents.sim_agents.simulation_router import discover_scale_agents

    data = {}
    for scale, info in discover_scale_agents().items():
        engines = info.get("supported", []) or []
        data[scale] = {
            eng: {"available": True, "source": "benchmark_mock"}
            for eng in engines
        }
    if not data:
        print("!! discover_scale_agents() returned empty — no skill bundles?",
              file=sys.stderr)
    return AvailableSoftware(data=data)


# ──────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────

def _score_one(query: RouterQuery, decision: dict,
               available_scales: set[str]) -> CaseResult:
    """Return a CaseResult given the router's decision.

    Three classes of result:
      * ``capability_gap`` — query.requires_agent names a scale that isn't
        in available_scales (e.g. molecular_dft today).  Not the router's
        fault; reported separately and excluded from accuracy.
      * ``error``          — router itself raised / returned an error key.
      * normal             — score against expected_scale / acceptable_*.
    """
    actual_scale  = decision.get("scale")
    actual_engine = decision.get("engine")
    error         = decision.get("error")

    # ── capability gap ───────────────────────────────────────────
    if query.requires_agent and query.requires_agent not in available_scales:
        return CaseResult(
            id=query.id,
            expected={"scale": query.expected_scale,
                      "engines": query.acceptable_engines,
                      "difficulty": query.difficulty,
                      "requires_agent": query.requires_agent},
            actual={"scale": actual_scale, "engine": actual_engine,
                    "router_returned": "None" if error else "fallback"},
            score=0.0,
            passed=False,
            notes=(f"capability gap: '{query.requires_agent}' agent not "
                   f"in the codebase; query excluded from accuracy"),
        )

    # ── router error ─────────────────────────────────────────────
    if error:
        return CaseResult(
            id=query.id,
            expected={"scale": query.expected_scale,
                      "engines": query.acceptable_engines},
            actual={"scale": None, "engine": None, "error": error},
            score=0.0,
            passed=False,
            notes=f"router error: {error}",
        )

    # ── normal scoring ───────────────────────────────────────────
    acceptable_scales = (
        list(query.acceptable_scales)
        if query.acceptable_scales
        else [query.expected_scale]
    )
    scale_ok = actual_scale in acceptable_scales
    engine_ok = (
        actual_engine in query.acceptable_engines
        if query.acceptable_engines
        else True
    )
    # Joint score: 1.0 if both, 0.5 if only scale, 0 otherwise.
    if scale_ok and engine_ok:
        score, passed = 1.0, True
    elif scale_ok:
        score, passed = 0.5, False
    else:
        score, passed = 0.0, False

    return CaseResult(
        id=query.id,
        expected={"scale": query.expected_scale,
                  "acceptable_scales": acceptable_scales,
                  "engines": query.acceptable_engines,
                  "difficulty": query.difficulty},
        actual={"scale": actual_scale,
                "engine": actual_engine,
                "reason": decision.get("reason", "")},
        score=score,
        passed=passed,
        notes=query.notes,
    )


# ──────────────────────────────────────────────────────────────────
#  Aggregate metrics  —  rolled into manifest.metrics
# ──────────────────────────────────────────────────────────────────

def _summary_metrics(cases: List[CaseResult],
                     queries: List[RouterQuery],
                     available_scales: set[str]) -> dict:
    """Accuracy is computed over scorable queries only — capability gaps
    are tallied separately so they don't drag the agent's score down."""
    n_total = len(cases)
    if n_total == 0:
        return {}

    # Partition: capability gaps vs. scorable
    gap_idx = [
        i for i, q in enumerate(queries)
        if q.requires_agent and q.requires_agent not in available_scales
    ]
    scorable_idx = [i for i in range(n_total) if i not in set(gap_idx)]
    n = len(scorable_idx)

    def _scale_hit(i: int) -> bool:
        q = queries[i]
        acc = q.acceptable_scales or [q.expected_scale]
        return cases[i].actual.get("scale") in acc

    def _engine_hit(i: int) -> bool:
        q = queries[i]
        if not q.acceptable_engines:
            return True
        return cases[i].actual.get("engine") in q.acceptable_engines

    out: dict = {
        "n_total":           n_total,
        "n_scorable":        n,
        "n_capability_gaps": len(gap_idx),
        "capability_gaps":   [queries[i].id for i in gap_idx],
    }
    if n:
        out["scale_accuracy"]  = sum(_scale_hit(i)  for i in scorable_idx) / n
        out["engine_accuracy"] = sum(_engine_hit(i) for i in scorable_idx) / n
        out["joint_accuracy"]  = sum(cases[i].passed for i in scorable_idx) / n
        for diff in ("easy", "medium", "hard"):
            ix = [i for i in scorable_idx if queries[i].difficulty == diff]
            if ix:
                out[f"joint_accuracy_{diff}"] = (
                    sum(cases[i].passed for i in ix) / len(ix)
                )
    return out


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_router", description=__doc__)
    parser.add_argument("--model",    default="claude-sonnet-4-5")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit",    type=int, default=None,
                        help="stop after this many queries (for fast probes)")
    parser.add_argument("--scale",    default=None,
                        help="only score queries with this expected scale")
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--out-dir",  default="benchmark/outputs/test_router")
    parser.add_argument("--dry-run",  action="store_true",
                        help="don't call the LLM; print what would be tested")
    parser.add_argument("--use-real-software", action="store_true",
                        help="use AvailableSoftware.auto() — the candidate "
                             "filter then reflects whatever's installed on "
                             "this machine.  Default is a mock that says "
                             "every agent-supported engine is available so "
                             "the LLM is scored on decision quality alone.")
    args = parser.parse_args(argv)

    queries: List[RouterQuery] = list(ALL_QUERIES)
    if args.scale:
        queries = [q for q in queries if q.expected_scale == args.scale]
    if args.difficulty:
        queries = [q for q in queries if q.difficulty == args.difficulty]
    if args.limit:
        queries = queries[: args.limit]

    print(f"router test :: {len(queries)} queries  model={args.model}")
    if args.dry_run:
        for q in queries:
            print(f"  [{q.difficulty:<6}] {q.id}  →  expect "
                  f"{q.expected_scale!r}  one_of={q.acceptable_engines}")
        return 0

    # Build router (lazy import so dry-run works without API key).
    from scilink.agents.sim_agents.simulation_router import (
        SimulationRouter, discover_scale_agents,
    )
    model = _build_model(args.model, args.api_key, args.base_url)
    if args.use_real_software:
        print("router :: using AvailableSoftware.auto() (real installed set)")
        router = SimulationRouter(model=model)
        available_scales = set(discover_scale_agents().keys())
    else:
        mock = _build_mock_software()
        scales = sorted(mock._data.keys())
        print(f"router :: using mock software (every engine available, "
              f"scales={scales})")
        router = SimulationRouter(model=model, available_software=mock)
        available_scales = set(mock._data.keys())

    manifest = RunnerManifest(runner="test_router", mode="local")
    for q in queries:
        try:
            decision = router.route(q.prompt)
        except Exception as exc:
            decision = {"error": f"router raised: {exc!r}"}
        case = _score_one(q, decision, available_scales)
        manifest.append(case)
        if "capability gap" in case.notes:
            flag = "·"     # not a router miss — tallied separately
        elif case.passed:
            flag = "✓"
        elif case.score > 0:
            flag = "~"
        else:
            flag = "✗"
        print(f"  {flag} [{q.difficulty:<6}] {q.id:<26} "
              f"got {case.actual.get('scale')!r:>34}/{case.actual.get('engine')!r:<12}"
              f"  score={case.score}")

    manifest.metrics = _summary_metrics(manifest.cases, queries, available_scales)
    out_dir = args.out_dir
    write_manifest(out_dir, manifest)
    extra_lines = ["\n## Per-difficulty accuracy (scorable only)\n"]
    for d in ("easy", "medium", "hard"):
        if f"joint_accuracy_{d}" in manifest.metrics:
            extra_lines.append(
                f"- `{d}`: {manifest.metrics[f'joint_accuracy_{d}']:.2f}"
            )
    if manifest.metrics.get("capability_gaps"):
        extra_lines.append("\n## Capability gaps (excluded from accuracy)\n")
        for cg in manifest.metrics["capability_gaps"]:
            extra_lines.append(f"- `{cg}`")
    write_summary_md(out_dir, manifest, extra="\n".join(extra_lines))

    print(f"\nwrote {out_dir}/manifest.json + summary.md")
    n_scorable = manifest.metrics.get("n_scorable", 0)
    n_gaps     = manifest.metrics.get("n_capability_gaps", 0)
    print(f"scorable: {n_scorable}   capability gaps: {n_gaps}")
    if n_scorable:
        print(f"scale accuracy:  {manifest.metrics['scale_accuracy']:.2f}")
        print(f"engine accuracy: {manifest.metrics['engine_accuracy']:.2f}")
        print(f"joint accuracy:  {manifest.metrics['joint_accuracy']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
