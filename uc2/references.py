"""Measured reference data for UC2 — 1 M Zn(OTf)2 in H2O / EIS mixtures.

EIS = ethyl isopropyl sulfone. The series varies the solvent volume ratio
(H2O:EIS) at fixed 1.0 M zinc triflate, all measured at 298.15 K (25 degrees C).

Two roles:
  * density + shear viscosity are the VALIDATION observables — computed from MD
    and checked against these measured values before any prediction is trusted;
  * the water-peak 1H T1 is the PREDICTION target — validated on a different
    observable (density/viscosity) than the one predicted, so the check is not
    circular.

Sources (local, not in the repo):
  ~/Documents/Projects/SciLink/SciLink_Dev/MD_Agents/NMR_Systems/DTN_ZnOTf2_H2O_EIS
    Zinc_electrolyte_Viscosity and Density.xls   -> density + viscosity vs T
    20250530_DTN_S*_...EIST1.json                -> 1H T1 per peak
"""

# Fixed measurement conditions for the series.
TEMPERATURE_K = 298.15
PRESSURE_ATM = 1.0
SALT = "zinc triflate, Zn(OTf)2"
SALT_CONCENTRATION_M = 1.0

# One entry per composition. `water_ratio` / `eis_ratio` are the solvent volume
# fractions the sample was mixed at. `viscosity_mPa_s` and `density_g_cm3` are
# the 25 C measured values (validation targets). `water_T1_s` is the measured 1H
# T1 of the water resonance at this composition (prediction target); `notes`
# flags where the assignment is unreliable.
COMPOSITIONS = [
    {
        "label": "80-20",
        "water_ratio": 80, "eis_ratio": 20,
        "viscosity_mPa_s": 2.7002,
        "density_g_cm3": 1.222518,
        "water_T1_s": 0.0401, "water_ppm": 5.125,
        "notes": "",
    },
    {
        "label": "70-30",
        "water_ratio": 70, "eis_ratio": 30,
        "viscosity_mPa_s": 3.4301,
        "density_g_cm3": 1.231321,
        "water_T1_s": 0.0337, "water_ppm": 4.85,
        "notes": "",
    },
    {
        "label": "60-40",
        "water_ratio": 60, "eis_ratio": 40,
        "viscosity_mPa_s": 4.4912,
        "density_g_cm3": 1.238860,
        "water_T1_s": 0.0277, "water_ppm": 4.538,
        "notes": "",
    },
    {
        "label": "50-50",
        "water_ratio": 50, "eis_ratio": 50,
        "viscosity_mPa_s": 5.7934,
        "density_g_cm3": 1.246252,
        # The 50-50 water resonance overlaps and its reported T1 (0.6317 s at
        # 4.24 ppm, SD 0.02 -- an order of magnitude larger relative error than
        # the others) does not follow the 0.0401 -> 0.0337 -> 0.0277 trend of
        # the lower-EIS points. Treat as unreliable; extrapolation of the trend
        # points to ~0.022 s.
        "water_T1_s": None, "water_ppm": 4.24,
        "notes": "water T1 assignment unreliable (overlap); excluded from the "
                 "T1 validation, kept for density/viscosity only",
    },
]

# The measured trend the whole use case turns on: from 80-20 to 50-50 BOTH
# viscosity and density RISE with EIS fraction. (An autonomous literature
# retrieval once returned the density direction inverted; this in-house series
# is the ground truth.)
MEASURED_TREND = {
    "viscosity": "increases with EIS fraction",
    "density": "increases with EIS fraction",
    "water_T1": "decreases with EIS fraction (faster relaxation, slower tumbling)",
}


def by_label(label):
    for c in COMPOSITIONS:
        if c["label"] == label:
            return c
    raise KeyError(f"no composition {label!r}; have "
                   f"{[c['label'] for c in COMPOSITIONS]}")


def labels():
    return [c["label"] for c in COMPOSITIONS]
