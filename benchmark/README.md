# SciLink agent benchmark suite

End-to-end tests of every simulation-side agent against
PNNL-relevant systems and workflows. Replaces the narrow
lattice-constant panel in `validation/` for the purpose of
*"how does the whole stack behave on real problems"*.

## What's tested

One runner per agent class. Each runner is self-contained,
scores into a shared format, and writes into `outputs/<runner>/`.

| Runner | Agent under test | Scope | Compute |
|---|---|---|---|
| `test_router.py` | `SimulationRouter` | 20–25 natural-language prompts → (scale, engine) labels | local |
| `test_structure_gen.py` | `StructureGenerator` | 10–12 systems from `systems.py`; spglib + validator checks | local |
| `test_quality.py` | `VaspQualityAgent` | curated outputs in `fixtures/vasp/` — known-good + planted issues | local |
| `test_dft.py` | `PeriodicDFTAgent` | systems.py DFT tier; full relax + property extraction | cluster |
| `test_updater.py` | `VaspUpdater` | planted error classes (fold-in of `examples/breakage_benchmark_*`) | cluster |
| `test_md.py` | `MDSimulationAgent` + `ForceFieldAgent` + `PackmolGen` + `LAMMPSAnalysisAgent` | water, NaCl(aq), LiPF₆/EC, glycine/H₂O | cluster |
| `test_mlip.py` | `MLIPAgent` + `MDSimulationAgent` (DeployedPotential handoff) | water (MACE), BCC Fe (MACE), LiCoO₂ (MACE) | cluster (GPU) |
| `test_orchestrator.py` | `SimulationOrchestratorAgent` | 3 chat sessions, one per PNNL theme | cluster + manual |

## PNNL focus areas covered

1–2 systems each:

- **Energy storage** — LiCoO₂ (cathode), LiPF₆ in EC (electrolyte)
- **Catalysis** — Pt(111) + CO@top, anatase TiO₂(101) + H₂O
- **Environmental / actinides** — UO₂ bulk (+U, f-electron stress test)
- **Aqueous + biomolecular** — water (TIP3P + MACE-MP-0), glycine zwitterion in water

Plus a baseline tier carried over from `validation/`:
BCC Fe, FCC Cu, Si diamond, C diamond, MgO rocksalt.

## Layout

```
benchmark/
├── README.md            ← this file
├── __init__.py
├── systems.py           ← reference-system registry  (load via get_system)
├── queries.py           ← router test prompts + ground-truth labels
├── _score.py            ← shared scoring + report-row utilities
├── runner.py            ← central CLI:  python -m benchmark.runner <test>
│
├── test_router.py
├── test_structure_gen.py
├── test_quality.py
├── test_dft.py
├── test_updater.py
├── test_md.py
├── test_mlip.py
├── test_orchestrator.py
│
├── aggregate.py         ← pulls every test's results → single report
│
├── fixtures/            ← committed inputs (planted-error VASP runs,
│                          known-good outputs for quality test, …)
└── outputs/             ← runtime artifacts (NOT committed — .gitignore)
    └── <test>/
        └── <system>/
            ├── manifest.json
            ├── inputs/
            └── results/
```

## Scoring contract

Every runner writes `outputs/<runner>/manifest.json` and a
`outputs/<runner>/summary.md`.  `aggregate.py` reads those and
produces a single `outputs/report.md` with one section per agent.

`manifest.json` shape:

```json
{
  "runner": "test_router",
  "mode": "local",
  "n_cases": 23,
  "passed": 19,
  "failed": 4,
  "metrics": {"scale_accuracy": 0.87, "engine_accuracy": 0.78},
  "cases": [ {"id": "...", "expected": {...}, "actual": {...}, "score": 1.0}, ... ]
}
```

## Schedule

| Day | Goal |
|---|---|
| Fri (today) | Skeleton + systems.py + queries.py drafted |
| Sat | Local tests complete (router, structure-gen, quality). Cluster harnesses drafted. |
| Sun | All cluster harnesses ready; dry-run locally |
| Mon | Submit cluster jobs; confirm local tests green |
| Tue–Wed | Cluster results stream in; aggregate report fills up |
| Thu+ | Docs + tutorials |

## Conventions

- One agent under test per runner. No cross-tests in one file.
- Runners take `--system <name>` / `--limit N` / `--dry-run` so we can
  iterate on a subset before sweeping the full roster.
- Each runner's `outputs/<runner>/` is the only thing it writes — never
  scribbles in the repo root or anywhere else.
- The PNNL system roster is the headline; the baseline tier is for
  regression and continuity with `validation/`.
