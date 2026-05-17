"""INCAR-generation stability test — same shape as test_router_variability,
applied one stage downstream of the router.

For each (model, prompt), fire ``PeriodicDFTAgent.generate_inputs`` N times
and measure:

  * **typo rate**  — fraction of trials whose *pre-fix* INCAR carries at
    least one bad tag according to pymatgen's ``Incar.check_params()``.
    Reads from ``result["syntax_check"]["issues"]`` — the pre-submit hook
    landed in 6d7206c auto-corrects high-confidence typos in place, but
    the raw issue list is preserved so we can score the agent's
    underlying quality independent of the validator's clean-up.

  * **physics stability** — analog of the router's "1.00 = identical pick
    every trial".  Each generated INCAR is reduced to a canonical tuple
    of physics-critical tags (ISPIN, ISIF, ISMEAR, ENCUT-band, NSW>0,
    MAGMOM-set, LDAU-set, IDIPOL/LDIPOL-set).  Stability = fraction of
    trials matching the modal tuple.

Prompts target physics-tricky systems where tag choices actually carry
information (vanilla Si bulk relax has one obvious answer; Fe / UO₂ /
Pt(111)+CO have several plausible setups and a few foot-guns).

Examples
--------
    python -m benchmark.runner test_incar_variability                # 8 trials × 3 prompts
    python -m benchmark.runner test_incar_variability --n-trials 4
    python -m benchmark.runner test_incar_variability \\
        --model claude-opus-4-7
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .systems import get_system
from .test_router import _build_model, _model_slug


# ──────────────────────────────────────────────────────────────────
#  Prompts
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IncarPrompt:
    id: str
    system_name: str            # key into benchmark.systems
    request: str                # natural-language goal
    difficulty: str             # easy | medium | hard
    physics_focus: str          # what makes this prompt foot-gunny


_DEFAULT_PROMPTS: Tuple[IncarPrompt, ...] = (
    IncarPrompt(
        id="fe_bcc_magnetic",
        system_name="fe_bcc",
        request=(
            "Set up a cell + ionic relaxation of BCC iron to obtain the "
            "equilibrium lattice constant.  Apply appropriate settings "
            "for a ferromagnetic metal."
        ),
        difficulty="easy",
        physics_focus="ISPIN + MAGMOM (Fe-specific; ISPN-for-ISPIN typo class)",
    ),
    IncarPrompt(
        id="uo2_dftU",
        system_name="uo2_fluorite",
        request=(
            "Set up a relaxation of UO₂ in the fluorite structure.  "
            "UO₂ is a strongly-correlated antiferromagnetic Mott "
            "insulator; treat the U 5f electrons appropriately."
        ),
        difficulty="hard",
        physics_focus="LDAUL / LDAUU / LDAUJ + ISPIN + MAGMOM-AFM",
    ),
    IncarPrompt(
        id="pt111_co_dipole",
        system_name="pt111_co_top",
        request=(
            "Set up a relaxation of CO adsorbed at the Pt(111) top site "
            "to compute the adsorption energy.  Account for the "
            "asymmetric slab geometry."
        ),
        difficulty="medium",
        physics_focus="IDIPOL / LDIPOL + ISMEAR for metal + selective dynamics",
    ),
)


# ──────────────────────────────────────────────────────────────────
#  Canonicalisation — reduce a raw INCAR text to its physics tuple
# ──────────────────────────────────────────────────────────────────

def _encut_band(encut: Optional[float]) -> str:
    if encut is None:
        return "unset"
    if encut < 400:  return "<400"
    if encut < 500:  return "400-500"
    if encut < 600:  return "500-600"
    return "≥600"


def _physics_tuple(incar_text: str) -> Tuple[Tuple[str, str], ...]:
    """Reduce an INCAR string to a tuple of (tag, canonical-value) pairs
    covering the physics-critical decisions.  Tag-typoed lines are
    invisible to ``Incar.from_str`` (pymatgen treats them as unknown
    keys), so a typo silently changes the tuple — which is exactly the
    behaviour we want this metric to surface."""
    try:
        from pymatgen.io.vasp.inputs import Incar
        incar = (Incar.from_str(incar_text) if hasattr(Incar, "from_str")
                 else Incar.from_string(incar_text))
    except Exception:
        return (("parse_error", "true"),)

    def _str(v):
        return str(v).strip() if v is not None else "unset"

    encut = incar.get("ENCUT")
    try:
        encut_f = float(encut) if encut is not None else None
    except (TypeError, ValueError):
        encut_f = None

    nsw_raw = incar.get("NSW", 0)
    try:
        nsw = int(nsw_raw)
    except (TypeError, ValueError):
        nsw = 0

    pairs = [
        ("ISPIN",   _str(incar.get("ISPIN", "unset"))),
        ("ISIF",    _str(incar.get("ISIF",  "unset"))),
        ("ISMEAR",  _str(incar.get("ISMEAR", "unset"))),
        ("ENCUT",   _encut_band(encut_f)),
        ("NSW>0",   "true" if nsw > 0 else "false"),
        ("MAGMOM",  "set" if incar.get("MAGMOM") is not None else "unset"),
        ("LDAU",    "set" if str(incar.get("LDAU", "")).strip(".").upper()
                       in ("TRUE", "T") else "unset"),
        ("IDIPOL",  _str(incar.get("IDIPOL", "unset"))),
        ("LDIPOL",  "true" if str(incar.get("LDIPOL", "")).strip(".").upper()
                       in ("TRUE", "T") else "unset"),
    ]
    return tuple(sorted(pairs))


# ──────────────────────────────────────────────────────────────────
#  Run one prompt × N trials
# ──────────────────────────────────────────────────────────────────

def _run_trials(model, prompt: IncarPrompt, n: int,
                api_key: str, base_url: Optional[str],
                model_name: str) -> List[Dict[str, Any]]:
    """Fire ``n`` generate_inputs calls on the same prompt."""
    from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent
    from ._vasp import write_poscar

    system = get_system(prompt.system_name)
    if system.fragments:
        # For multi-fragment cells (adsorption), drive the "full" build
        # which is the structurally interesting one.  Single-cell
        # generation per prompt keeps the test tight.
        atoms = system.fragments["full"]()
    else:
        atoms = system.build()

    trials: List[Dict[str, Any]] = []
    for i in range(n):
        with tempfile.TemporaryDirectory(prefix=f"incarvar_{prompt.id}_") as td:
            poscar = os.path.join(td, "POSCAR")
            write_poscar(atoms, poscar)
            agent = PeriodicDFTAgent(
                api_key=api_key, base_url=base_url, model_name=model_name)
            try:
                result = agent.generate_inputs(
                    structure_file=poscar,
                    request=prompt.request,
                    software="vasp",
                )
            except Exception as exc:
                trials.append({
                    "trial": i + 1,
                    "error": f"generate_inputs raised: {exc!r}",
                })
                continue

            if result.get("status") != "success":
                trials.append({
                    "trial": i + 1,
                    "error": result.get("message", "generate_inputs failed"),
                })
                continue

            incar_text = result["input_files"].get("INCAR", "")
            sc = result.get("syntax_check", {}) or {}
            pre_fix_issues = sc.get("issues", []) or []
            applied_fixes = sc.get("applied_fixes", []) or []
            phys = _physics_tuple(incar_text)

            trials.append({
                "trial":            i + 1,
                "incar":            incar_text,
                "physics_tuple":    list(phys),
                "n_pre_fix_issues": len(pre_fix_issues),
                "pre_fix_issues":   pre_fix_issues,
                "n_applied_fixes":  len(applied_fixes),
                "applied_fixes":    applied_fixes,
                "error":            None,
            })
    return trials


def _summarise_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [t for t in trials if not t.get("error")]
    n  = len(ok)
    if n == 0:
        return {
            "n_trials": 0, "n_errors": len(trials), "typo_rate": None,
            "physics_stability": None, "modal_tuple": None,
        }

    n_with_typo = sum(1 for t in ok if t["n_pre_fix_issues"] > 0)
    typo_rate = round(n_with_typo / n, 3)

    typo_tags: Counter = Counter()
    for t in ok:
        for issue in t["pre_fix_issues"]:
            tag = issue.get("tag") or "(unknown)"
            typo_tags[tag] += 1

    tuples = [tuple(map(tuple, t["physics_tuple"])) for t in ok]
    tup_counts = Counter(tuples)
    modal_tuple, modal_n = tup_counts.most_common(1)[0]
    physics_stability = round(modal_n / n, 3)

    return {
        "n_trials":          n,
        "n_errors":          len(trials) - n,
        "typo_rate":         typo_rate,
        "typo_tags":         dict(typo_tags.most_common()),
        "physics_stability": physics_stability,
        "distinct_tuples":   len(tup_counts),
        "modal_tuple":       [list(p) for p in modal_tuple],
    }


# ──────────────────────────────────────────────────────────────────
#  Markdown summary
# ──────────────────────────────────────────────────────────────────

def _render_summary_md(model: str, n_trials: int,
                       per_prompt: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# test_incar_variability :: {model}")
    lines.append("")
    lines.append(f"- trials per prompt: **{n_trials}**")
    lines.append(f"- prompts: {len(per_prompt)}")

    typo_rates = [p["summary"]["typo_rate"]
                  for p in per_prompt.values()
                  if p["summary"]["typo_rate"] is not None]
    phys_stabs = [p["summary"]["physics_stability"]
                  for p in per_prompt.values()
                  if p["summary"]["physics_stability"] is not None]
    if typo_rates:
        lines.append(f"- mean typo rate: **{sum(typo_rates)/len(typo_rates):.2f}** "
                     f"(fraction of trials whose pre-fix INCAR carried ≥1 unknown tag)")
    if phys_stabs:
        lines.append(f"- mean physics stability: **{sum(phys_stabs)/len(phys_stabs):.2f}** "
                     f"(1.00 = identical physics-tuple every trial)")
    lines.append("")

    for pid, body in per_prompt.items():
        p = body["prompt"]
        s = body["summary"]
        lines.append(f"## {pid}  ({p['difficulty']})")
        lines.append("")
        lines.append(f"> {p['request']}")
        lines.append("")
        lines.append(f"- physics focus: *{p['physics_focus']}*")
        lines.append(f"- trials OK / errored: **{s['n_trials']} / {s['n_errors']}**")
        if s["typo_rate"] is not None:
            lines.append(f"- typo rate: **{s['typo_rate']:.2f}**")
            if s.get("typo_tags"):
                rows = ", ".join(f"`{t}` × {n}" for t, n in s["typo_tags"].items())
                lines.append(f"  - bad tags observed: {rows}")
        if s["physics_stability"] is not None:
            lines.append(f"- physics stability: **{s['physics_stability']:.2f}** "
                         f"({s['distinct_tuples']} distinct "
                         f"tuple{'s' if s['distinct_tuples'] > 1 else ''} "
                         f"across {s['n_trials']} trials)")
        if s.get("modal_tuple"):
            lines.append("")
            lines.append("Modal physics-tuple:")
            lines.append("")
            lines.append("| tag | canonical value |")
            lines.append("|---|---|")
            for tag, val in s["modal_tuple"]:
                lines.append(f"| `{tag}` | `{val}` |")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def _pick_prompts(ids: List[str]) -> List[IncarPrompt]:
    by_id = {p.id: p for p in _DEFAULT_PROMPTS}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"!! unknown prompt id(s): {missing}\n"
                         f"   known: {sorted(by_id)}")
    return [by_id[i] for i in ids]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="test_incar_variability", description=__doc__)
    parser.add_argument("--model",     default="claude-sonnet-4-5")
    parser.add_argument("--api-key",   default=None)
    parser.add_argument("--base-url",  default=None)
    parser.add_argument("--n-trials",  type=int, default=8,
                        help="generate_inputs calls per prompt (default 8)")
    parser.add_argument("--prompts",
                        default=",".join(p.id for p in _DEFAULT_PROMPTS),
                        help="comma-separated prompt ids")
    parser.add_argument("--out-dir",   default=None,
                        help="output directory.  Default auto-namespaces "
                             "under benchmark/outputs/test_incar_variability/"
                             "<model-slug>/")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args(argv)

    pick_ids = [p.strip() for p in args.prompts.split(",") if p.strip()]
    prompts = _pick_prompts(pick_ids)

    print(f"test_incar_variability :: {len(prompts)} prompts × "
          f"{args.n_trials} trials  model={args.model}")
    for p in prompts:
        print(f"  [{p.difficulty:<6}] {p.id:<22}  focus={p.physics_focus}")

    if args.dry_run:
        return 0

    api_key = args.api_key or os.environ.get("SCILINK_API_KEY")
    base_url = args.base_url or os.environ.get("SCILINK_BASE_URL")
    if not api_key:
        raise SystemExit("no API key — set SCILINK_API_KEY or pass --api-key")

    # _build_model is reused for symmetry with test_router_variability
    # but PeriodicDFTAgent builds its own model from (api_key, base_url,
    # model_name); we don't need the model object here.  We do call it
    # to fail-fast if the wiring is wrong.
    _ = _build_model(args.model, api_key, base_url)

    per_prompt: Dict[str, Dict[str, Any]] = {}
    for prompt in prompts:
        print(f"\n— {prompt.id} ({prompt.difficulty}) —")
        trials = _run_trials(
            model=None, prompt=prompt, n=args.n_trials,
            api_key=api_key, base_url=base_url, model_name=args.model)
        summary = _summarise_trials(trials)
        per_prompt[prompt.id] = {
            "prompt": {
                "id":            prompt.id,
                "request":       prompt.request,
                "difficulty":    prompt.difficulty,
                "physics_focus": prompt.physics_focus,
                "system":        prompt.system_name,
            },
            "trials":  trials,
            "summary": summary,
        }
        for t in trials:
            if t.get("error"):
                print(f"  ✗ trial {t['trial']:2d}  error: {t['error']}")
            else:
                fix_note = (f"  (fixed {t['n_applied_fixes']})"
                            if t["n_applied_fixes"] else "")
                print(f"  {'✓' if t['n_pre_fix_issues']==0 else '!'} "
                      f"trial {t['trial']:2d}  "
                      f"pre-fix typos: {t['n_pre_fix_issues']}{fix_note}")
        if summary["n_trials"]:
            print(f"  → typo rate {summary['typo_rate']:.2f}   "
                  f"physics stability {summary['physics_stability']:.2f}   "
                  f"({summary['distinct_tuples']} distinct "
                  f"tuple{'s' if summary['distinct_tuples'] > 1 else ''})")

    # ── Persist ─────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.join(
        "benchmark/outputs/test_incar_variability", _model_slug(args.model))
    os.makedirs(out_dir, exist_ok=True)

    typo_rates = [p["summary"]["typo_rate"]
                  for p in per_prompt.values()
                  if p["summary"]["typo_rate"] is not None]
    phys_stabs = [p["summary"]["physics_stability"]
                  for p in per_prompt.values()
                  if p["summary"]["physics_stability"] is not None]
    payload = {
        "runner":   "test_incar_variability",
        "mode":     "local",
        "model":    args.model,
        "n_trials": args.n_trials,
        "prompts":  per_prompt,
        "metrics": {
            "model":                args.model,
            "n_trials":             args.n_trials,
            "n_prompts":            len(per_prompt),
            "mean_typo_rate":       round(sum(typo_rates)/len(typo_rates), 3)
                                    if typo_rates else None,
            "mean_physics_stability": round(sum(phys_stabs)/len(phys_stabs), 3)
                                      if phys_stabs else None,
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(payload, f, indent=2)

    md = _render_summary_md(args.model, args.n_trials, per_prompt)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)

    print(f"\nwrote {out_dir}/manifest.json + summary.md")
    if typo_rates:
        print(f"mean typo rate: {payload['metrics']['mean_typo_rate']:.2f}")
    if phys_stabs:
        print(f"mean physics stability: {payload['metrics']['mean_physics_stability']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
