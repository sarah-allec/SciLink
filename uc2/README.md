# UC2 — experiment-validated prediction for a Zn(OTf)₂ / H₂O / EIS electrolyte

The use case: 1 M zinc triflate in a water / ethyl-isopropyl-sulfone (EIS) mixed
solvent, across four solvent ratios (H₂O:EIS from 80:20 to 50:50) at 298.15 K.
The point is **not** to predict blind — it is to *validate against measured data
first*, catch where the force field fails, and only then trust a prediction:

1. Compute **density** and **shear viscosity** (Green-Kubo) per composition and
   compare to Dave's measured values.
2. If the model reproduces them, the **¹H water-T1 prediction** is trustworthy;
   if it doesn't (e.g. the viscosity trend comes out backwards), that is the
   force-field limitation to fix before predicting.

Density + viscosity are validated; T1 is predicted — a *different* observable
than the one validated, so the check isn't circular.

## Files

| file | role |
|---|---|
| `references.py` | measured density / viscosity / water-T1 at 298 K, per composition; the ground-truth trend |
| `run_uc2.py` | the driver — `--stage {preview,full,analyze,validate}` |
| `submit_uc2.sbatch` | Deception GPU array job, one member per task |
| `validate_uc2.py` | computed-vs-measured table + trend verdict |

Measured data itself lives outside the repo:
`~/Documents/Projects/SciLink/SciLink_Dev/MD_Agents/NMR_Systems/DTN_ZnOTf2_H2O_EIS`.

## Running it

**1. Review the generation interactively (login node).** One member is enough to
sanity-check the approach — box, force field, and the derived observable/sampling
contract — with human feedback before spending GPU time:

```bash
python uc2/run_uc2.py --stage preview --members 80-20
```

Nothing runs; it generates and validates the deck and stops. Inspect
`uc2/runs/80-20/` and the dry-run gate output. Iterate until it looks right.

**2. Run the series (compute nodes).** Set the environment (see below), then:

```bash
sbatch uc2/submit_uc2.sbatch            # all four members (array 0-3)
# or a single member:
sbatch --array=2 uc2/submit_uc2.sbatch  # 60-40
```

Each array task generates + runs + refines one member autonomously, then analyzes
its own output.

**3. Validate (login node), once all four finish:**

```bash
python uc2/run_uc2.py --stage validate
```

## Environment

The driver reads credentials and run parameters from the environment:

| var | meaning |
|---|---|
| `SCILINK_BASE_URL` | proxy URL. When set, auth uses `SCILINK_API_KEY`; model names must be the `-project` variants. |
| `SCILINK_API_KEY` | proxy key (with `SCILINK_BASE_URL`) |
| `ANTHROPIC_API_KEY` | direct path, when no `SCILINK_BASE_URL` |
| `SCILINK_MODEL` | default `claude-opus-4-8-project` |
| `FUTUREHOUSE_API_KEY` | optional, literature-grounded steps |
| `LAMMPS_CMD` | run-command template, must contain `{script}` (default `lmp -in {script}`) |
| `UC2_ENV_SETUP` | path to the env-activation script the sbatch sources (the `scilink_ffmd` OpenFF + LAMMPS env; also needs its conda libs on `LD_LIBRARY_PATH` and numpy<2.3) |
| `UC2_STAGE_TIMEOUT` | per-stage wall-clock cap, seconds |

## Notes

- The whole use case turns on one measured fact: from 80:20 to 50:50 **both
  viscosity and density rise** with EIS fraction. An autonomous literature
  retrieval once returned the density direction inverted — the in-house series is
  the ground truth, not the literature.
- The 50:50 water-T1 is excluded from the T1 comparison (overlapping resonance,
  unreliable fit); it still contributes density + viscosity.
- Generation is meant to be reviewed by a human (`preview`), because for the use
  case the science has to be right — this is not a throughput benchmark.
