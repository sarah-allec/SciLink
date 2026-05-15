"""Shared scoring + manifest utilities.

Every test runner uses the same manifest shape so aggregate.py can read
them uniformly.  See benchmark/README.md for the contract.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CaseResult:
    """One scored case (one prompt, one system, one error class …)."""
    id: str
    expected: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    notes: str = ""


@dataclass
class RunnerManifest:
    """What each test runner writes to outputs/<runner>/manifest.json."""
    runner: str
    mode: str                    # "local" | "cluster"
    n_cases: int = 0
    passed: int = 0
    failed: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    cases: List[CaseResult] = field(default_factory=list)

    def append(self, case: CaseResult) -> None:
        self.cases.append(case)
        self.n_cases = len(self.cases)
        if case.passed:
            self.passed += 1
        else:
            self.failed += 1

    def to_dict(self) -> dict:
        d = asdict(self)
        # CaseResult was already serialised by asdict
        return d


def write_manifest(out_dir: str, manifest: RunnerManifest) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return path


def write_summary_md(out_dir: str, manifest: RunnerManifest,
                     extra: str = "") -> str:
    """Short human-readable summary; aggregate.py just concatenates these."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.md")
    lines = [
        f"# {manifest.runner}",
        "",
        f"- mode: `{manifest.mode}`",
        f"- cases: **{manifest.n_cases}**  "
        f"(passed **{manifest.passed}**, failed **{manifest.failed}**)",
    ]
    if manifest.metrics:
        lines.append("- metrics:")
        for k, v in manifest.metrics.items():
            if isinstance(v, float):
                lines.append(f"  - `{k}` = {v:.3f}")
            else:
                lines.append(f"  - `{k}` = {v}")
    if extra:
        lines.append("")
        lines.append(extra)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
