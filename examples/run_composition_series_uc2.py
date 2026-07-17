"""UC2: aqueous Zn(OTf)2 solvation vs cosolvent fraction, as a controlled series.

Runs ONE MD protocol across four water:EIS cosolvent fractions (80:20 -> 50:50
by volume, all 1.0 M Zn(OTf)2), so the members differ only by composition and
the solvation trend across them is a controlled comparison — unlike the earlier
campaign, where each composition was generated independently (system size swung
221->1975 water, Zn count 8->45).

The composition arithmetic is explicit and parameterized below. RHO_EIS is the
one external datum; it is a documented estimate (the earlier independent runs
imply 0.96-1.23 g/cm^3), not a measured value — adjust it if a better number is
available, or bootstrap it from a pure-EIS NPT run.

Environment (same as the MD one-shot smoke):
  SCILINK_API_KEY, SCILINK_BASE_URL   proxy credentials
  SCILINK_MODEL                       optional model (default claude-opus-4-8-project)
  SCILINK_RUN_COMMAND                 optional 'lmp -in {script}' override

Usage:
  python examples/run_composition_series_uc2.py            # generate only
  python examples/run_composition_series_uc2.py --run      # generate + run + refine
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── composition parameters (edit these; the counts derive from them) ──
RHO_WATER = 0.997      # g/cm^3, 25 C
RHO_EIS = 1.10         # g/cm^3 — DOCUMENTED ESTIMATE, confirm/replace if known
MW_WATER = 18.015      # g/mol
MW_EIS = 136.21        # g/mol, ethyl isopropyl sulfone (C5H12O2S)

N_SOLVENT_TOTAL = 2000    # water + EIS per member — fixed for a consistent series
N_ZN = 40                 # fixed ion count for consistent statistics (was 8-45)
VOL_FRACTIONS = {         # water:EIS by volume -> EIS volume fraction
    "S2_80-20": 0.20, "S3_70-30": 0.30, "S4_60-40": 0.40, "S5_50-50": 0.50,
}
START_DENSITY = 1.20      # g/cm^3 starting guess for packing; NPT relaxes it

SMILES = {"water": "O", "eis": "CCS(=O)(=O)C(C)C",
          "zn": "[Zn+2]", "triflate": "[O-]S(=O)(=O)C(F)(F)F"}

GOAL = (
    "Run classical molecular dynamics of a 1.0 M zinc triflate (Zn(OTf)2) "
    "aqueous electrolyte with an ethyl isopropyl sulfone cosolvent at 298 K "
    "and 1 atm, equilibrating in NPT, to examine how the Zn2+ solvation "
    "structure (the species-resolved Zn-O radial distribution) changes with "
    "cosolvent fraction. Produce RDFs to examine the solvation structure."
)


def eis_water_counts(vol_frac_eis: float) -> tuple[int, int]:
    """Integer (N_water, N_EIS) for a target EIS volume fraction at fixed total.

    N_EIS/N_water = (v/(1-v)) * (rho_EIS*MW_water)/(rho_water*MW_EIS).
    """
    k = (RHO_EIS * MW_WATER) / (RHO_WATER * MW_EIS)
    ratio = (vol_frac_eis / (1.0 - vol_frac_eis)) * k
    n_water = round(N_SOLVENT_TOTAL / (1.0 + ratio))
    n_eis = N_SOLVENT_TOTAL - n_water
    return n_water, n_eis


def build_members() -> list:
    members = []
    for name, vfrac in VOL_FRACTIONS.items():
        n_water, n_eis = eis_water_counts(vfrac)
        members.append({
            "name": name,
            "density": START_DENSITY,
            "components": [
                {"name": "water", "smiles": SMILES["water"], "count": n_water},
                {"name": "ethyl_isopropyl_sulfone", "smiles": SMILES["eis"], "count": n_eis},
                {"name": "Zn2+", "smiles": SMILES["zn"], "count": N_ZN},
                {"name": "triflate", "smiles": SMILES["triflate"], "count": 2 * N_ZN},
            ],
        })
    return members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="execute + refine (needs an engine); else generate only")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    members = build_members()
    print("=== composition series ===")
    for m in members:
        c = {x["name"]: x["count"] for x in m["components"]}
        print(f"  {m['name']}: water={c['water']} eis={c['ethyl_isopropyl_sulfone']} "
              f"Zn={c['Zn2+']} triflate={c['triflate']}")
    print(f"(RHO_EIS={RHO_EIS} g/cm^3 — estimate)\n")

    from scilink.agents.sim_agents.simulation_pipeline import run_composition_series

    api_key = os.environ.get("SCILINK_API_KEY")
    base_url = os.environ.get("SCILINK_BASE_URL")
    model = os.environ.get("SCILINK_MODEL", "claude-opus-4-8-project")
    if not api_key or not base_url:
        sys.exit("Set SCILINK_API_KEY and SCILINK_BASE_URL (proxy) first.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.base_dir or f"./uc2_series_{stamp}").resolve()

    executor = run_command = None
    if args.run:
        from scilink.agents.sim_agents.refinement import LocalExecutor
        from scilink.skills._shared._registry import get_tool_function
        run_command = os.environ.get("SCILINK_RUN_COMMAND")
        if not run_command:
            get_rc = get_tool_function("default_run_command", active_skills=["lammps"])
            run_command = get_rc()
        if not run_command:
            sys.exit("No LAMMPS run command (set SCILINK_RUN_COMMAND).")
        executor = LocalExecutor(timeout=int(os.environ.get("LMP_TIMEOUT", "36000")))

    result = run_composition_series(
        GOAL, members, software="lammps", density=START_DENSITY,
        output_dir=str(base_dir), api_key=api_key, base_url=base_url,
        model_name=model, executor=executor, run_command=run_command,
        autonomy="autonomous",
    )

    print("\n=== RESULT ===")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("stages",)}, indent=2, default=str))
    print(f"\nSession dir: {base_dir}")


if __name__ == "__main__":
    main()
