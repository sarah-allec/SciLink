"""
Backend-agnostic tools for ML interatomic potentials.

Design: every public function accepts a `backend` parameter and dispatches
to private backend-specific implementations.  This keeps the agent code
clean while making it easy to add NequIP / DeePMD / Allegro later without
touching the agent.

The heavy MACE coverage reflects reality: MACE currently has the only
production-ready foundation model ecosystem.  NequIP and DeePMD dispatch
points raise NotImplementedError with actionable messages so failures are
obvious rather than silent.
"""

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  BACKEND AVAILABILITY
# ═══════════════════════════════════════════════════════════════════

def check_backends() -> Dict[str, Dict[str, Any]]:
    """
    Check which MLIP backends are installed.

    Returns dict keyed by backend name:
        { "mace": {"available": bool, "version": str, "pretrained": [...]}, ... }
    """
    result: Dict[str, Dict[str, Any]] = {}

    # ── MACE ──────────────────────────────────────────────────────
    try:
        import mace
        result["mace"] = {
            "available": True,
            "version": getattr(mace, "__version__", "unknown"),
            "lammps_pair_style": "mace",
            "pretrained": [
                {
                    "name": "mace-mp-0",
                    "domain": "inorganic",
                    "description": (
                        "Universal potential trained on MPtrj dataset "
                        "(~150k inorganic materials). Good for metals, "
                        "oxides, ceramics."
                    ),
                },
                {
                    "name": "mace-mp-0b",
                    "domain": "inorganic",
                    "description": "Medium-accuracy variant, faster inference.",
                },
                {
                    "name": "mace-off23",
                    "domain": "organic",
                    "description": (
                        "Trained on SPICE dataset. Covers organic molecules, "
                        "peptides, drug-like compounds."
                    ),
                },
            ],
        }
    except ImportError:
        result["mace"] = {
            "available": False,
            "version": None,
            "pretrained": [],
        }

    # ── NequIP ────────────────────────────────────────────────────
    try:
        import nequip
        result["nequip"] = {
            "available": True,
            "version": getattr(nequip, "__version__", "unknown"),
            "lammps_pair_style": "nequip",
            "pretrained": [],       # NequIP has no foundation models
        }
    except ImportError:
        result["nequip"] = {"available": False, "version": None, "pretrained": []}

    # ── DeePMD ────────────────────────────────────────────────────
    try:
        import deepmd
        result["deepmd"] = {
            "available": True,
            "version": getattr(deepmd, "__version__", "unknown"),
            "lammps_pair_style": "deepmd",
            "pretrained": [],       # DPA-2 may change this
        }
    except ImportError:
        result["deepmd"] = {"available": False, "version": None, "pretrained": []}

    return result


# ═══════════════════════════════════════════════════════════════════
#  PRETRAINED MODEL DEPLOYMENT  (the default path)
# ═══════════════════════════════════════════════════════════════════

def deploy_pretrained(
    backend: str,
    model_name: str,
    elements: List[str],
    working_dir: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Deploy a pretrained foundation model for immediate use.
    No training data or GPU training time required.

    Args:
        backend:    "mace" | "nequip" | "deepmd"
        model_name: Pretrained model identifier (e.g. "mace-mp-0")
        elements:   Chemical elements in the system
        working_dir: Output directory
        device:     "cpu" or "cuda"

    Returns:
        {
            "model_file": str,      # path to model artifact
            "calculator": object,   # ASE calculator (for validation)
            "metadata": dict,       # model metadata
        }
    """
    os.makedirs(working_dir, exist_ok=True)

    if backend == "mace":
        return _mace_deploy_pretrained(model_name, elements, working_dir, device)
    elif backend == "nequip":
        raise NotImplementedError(
            "NequIP has no foundation models. Use backend='mace' for "
            "pretrained deployment, or call train() with NequIP."
        )
    elif backend == "deepmd":
        raise NotImplementedError(
            "DeePMD pretrained deployment not yet implemented. "
            "DPA-2 foundation model support is planned."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _mace_deploy_pretrained(
    model_name: str,
    elements: List[str],
    working_dir: str,
    device: str,
) -> Dict[str, Any]:
    """Deploy a pretrained MACE model."""
    from mace.calculators import mace_mp, mace_off

    # ── Map human-readable names to MACE API arguments ────────────
    # mace_mp() expects: "small", "medium", "large", or a file path/URL
    # mace_off() expects: "small", "medium", "large", or a file path/URL
    MP_MODEL_MAP = {
        "mace-mp-0":   "medium",
        "mace-mp-0b":  "small",
        "mace-mp-0-large": "large",
        "small":       "small",
        "medium":      "medium",
        "large":       "large",
    }
    OFF_MODEL_MAP = {
        "mace-off23":       "medium",
        "mace-off23-small": "small",
        "mace-off23-large": "large",
    }

    is_off = "off" in model_name.lower()

    if is_off:
        mace_size = OFF_MODEL_MAP.get(model_name, model_name)
        calc = mace_off(model=mace_size, device=device, default_dtype="float64")
        domain = "organic"
    else:
        mace_size = MP_MODEL_MAP.get(model_name, model_name)
        calc = mace_mp(model=mace_size, device=device, default_dtype="float64")
        domain = "inorganic"

    # Locate the cached model file for LAMMPS deployment
    model_file = None
    for attr in ("model_path", "model_paths"):
        val = getattr(calc, attr, None)
        if val:
            model_file = str(val) if not isinstance(val, list) else str(val[0])
            break

    if model_file is None or not os.path.exists(model_file):
        model_file = os.path.join(working_dir, f"{model_name}.model")
        try:
            import torch
            torch.save(calc.models[0], model_file)
        except Exception as e:
            logger.warning(f"Could not save model file: {e}")
            model_file = model_name

    logger.info(f"Deployed pretrained {model_name} → {mace_size} ({domain}) → {model_file}")

    return {
        "model_file": model_file,
        "calculator": calc,
        "metadata": {
            "backend": "mace",
            "model_name": model_name,
            "mace_size": mace_size,
            "domain": domain,
            "device": device,
            "elements_requested": elements,
        },
    }

# ═══════════════════════════════════════════════════════════════════
#  UNCERTAINTY ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_uncertainty(
    backend: str,
    model_file: str,
    structures: List[Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate model uncertainty on a set of structures.

    For committee/ensemble models, uses prediction variance.
    For single models, uses energy-based heuristics.

    Args:
        backend:    MLIP backend name
        model_file: Path to model artifact
        structures: List of ase.Atoms objects

    Returns:
        {
            "per_structure": [
                {
                    "index": int,
                    "energy_uncertainty": float,    # meV/atom
                    "max_force_uncertainty": float,  # meV/Å
                    "is_extrapolation": bool,
                },
                ...
            ],
            "mean_energy_uncertainty": float,
            "max_energy_uncertainty": float,
            "n_extrapolating": int,
            "extrapolation_indices": [int],
        }
    """
    if backend == "mace":
        return _mace_evaluate_uncertainty(model_file, structures, device)
    else:
        raise NotImplementedError(
            f"Uncertainty estimation for {backend} not yet implemented."
        )


def _mace_evaluate_uncertainty(
    model_file: str,
    structures: List[Any],
    device: str,
) -> Dict[str, Any]:
    """
    MACE uncertainty via per-atom energy variance heuristic.

    For true uncertainty, use a committee of models.  This single-model
    approach flags structures where per-atom energies have unusual
    distributions compared to typical bulk configurations.
    """
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=model_file,
        device=device,
        default_dtype="float64",
    )

    per_structure = []
    energy_uncertainties = []

    for i, atoms in enumerate(structures):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()

            # Per-atom energy is available in MACE via the calculator
            e_per_atom = energy / len(atoms_copy)

            # Force magnitude distribution — high max forces suggest
            # the model is in an unfamiliar region
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = float(np.max(force_norms))

            # Heuristic uncertainty: large force variance + extreme
            # per-atom energy → likely extrapolation
            force_std = float(np.std(force_norms))

            # Convert to meV for consistency
            energy_unc = force_std * 1000      # rough proxy
            force_unc = max_force * 1000

            # Flag as extrapolation if max force > 10 eV/Å
            is_extrap = max_force > 10.0

            per_structure.append({
                "index": i,
                "energy_per_atom": float(e_per_atom),
                "energy_uncertainty": energy_unc,
                "max_force": max_force,
                "max_force_uncertainty": force_unc,
                "force_std": force_std,
                "is_extrapolation": is_extrap,
            })
            energy_uncertainties.append(energy_unc)

        except Exception as e:
            logger.warning(f"Uncertainty evaluation failed for structure {i}: {e}")
            per_structure.append({
                "index": i,
                "energy_uncertainty": float("inf"),
                "max_force_uncertainty": float("inf"),
                "is_extrapolation": True,
                "error": str(e),
            })
            energy_uncertainties.append(float("inf"))

    extrap_indices = [s["index"] for s in per_structure if s["is_extrapolation"]]

    return {
        "per_structure": per_structure,
        "mean_energy_uncertainty": float(np.mean(
            [e for e in energy_uncertainties if np.isfinite(e)]
        )) if energy_uncertainties else 0.0,
        "max_energy_uncertainty": float(np.max(
            [e for e in energy_uncertainties if np.isfinite(e)]
        )) if energy_uncertainties else 0.0,
        "n_extrapolating": len(extrap_indices),
        "extrapolation_indices": extrap_indices,
    }


# ═══════════════════════════════════════════════════════════════════
#  TRAINING DATASET
# ═══════════════════════════════════════════════════════════════════

def build_training_dataset(
    structures: List[Any],
    working_dir: str,
    existing_data_path: Optional[str] = None,
    max_structures: int = 500,
    val_fraction: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Build an extXYZ training dataset.  Backend-agnostic — all backends
    can read extXYZ.

    Returns:
        {
            "train_file": str, "val_file": str,
            "n_train": int, "n_val": int,
            "elements": [str],
            "energy_mean": float, "energy_std": float,
        }
    """
    import ase.io

    os.makedirs(working_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    # Collect frames
    frames = []
    if existing_data_path:
        logger.info(f"Loading dataset from {existing_data_path}")
        frames.extend(ase.io.read(existing_data_path, index=":"))
    for item in (structures or []):
        if isinstance(item, (str, Path)):
            frames.extend(ase.io.read(str(item), index=":"))
        else:
            frames.append(item)

    # Keep only frames with energy + forces
    valid = [
        f for f in frames
        if (f.calc is not None
            and "energy" in (f.calc.results or {})
            and "forces" in (f.calc.results or {}))
    ]
    logger.info(f"Valid frames (with energy+forces): {len(valid)}/{len(frames)}")

    if not valid:
        raise ValueError("No frames with energy+forces found.")

    # Subsample
    if len(valid) > max_structures:
        idx = rng.choice(len(valid), max_structures, replace=False)
        valid = [valid[i] for i in sorted(idx)]

    # Split
    rng.shuffle(valid)
    n_val = max(1, int(val_fraction * len(valid)))
    train, val = valid[n_val:], valid[:n_val]

    train_file = os.path.join(working_dir, "train.xyz")
    val_file = os.path.join(working_dir, "val.xyz")
    ase.io.write(train_file, train, format="extxyz")
    ase.io.write(val_file, val, format="extxyz")

    energies = [f.calc.results["energy"] / len(f) for f in valid]
    elements = sorted({s for f in valid for s in f.get_chemical_symbols()})

    return {
        "train_file": train_file,
        "val_file": val_file,
        "n_train": len(train),
        "n_val": len(val),
        "elements": elements,
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
    }


# ═══════════════════════════════════════════════════════════════════
#  TRAINING  (backend dispatch)
# ═══════════════════════════════════════════════════════════════════

def train(
    backend: str,
    dataset_info: Dict[str, Any],
    working_dir: str,
    foundation_model: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    timeout_hours: float = 12.0,
) -> Dict[str, Any]:
    """
    Train or fine-tune an MLIP.

    Args:
        backend: "mace" | "nequip" | "deepmd"
        dataset_info: Output of build_training_dataset()
        working_dir: Output directory
        foundation_model: If set, fine-tune from this checkpoint
        hyperparameters: Backend-specific training config overrides
        timeout_hours: Hard wall-clock limit

    Returns:
        { "model_file": str, "validation": dict, "status": str }
    """
    hparams = hyperparameters or {}

    if backend == "mace":
        return _mace_train(dataset_info, working_dir, foundation_model,
                           hparams, timeout_hours)
    elif backend == "nequip":
        raise NotImplementedError(
            "NequIP training: implement _nequip_train() using nequip-train CLI "
            "and a YAML config.  The interface mirrors _mace_train()."
        )
    elif backend == "deepmd":
        raise NotImplementedError(
            "DeePMD training: implement _deepmd_train() using dp train CLI "
            "and a JSON config.  The interface mirrors _mace_train()."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _mace_train(
    dataset_info: Dict[str, Any],
    working_dir: str,
    foundation_model: Optional[str],
    hparams: Dict[str, Any],
    timeout_hours: float,
) -> Dict[str, Any]:
    """MACE training via mace_run_train CLI."""
    model_dir = os.path.join(working_dir, "mace_model")
    os.makedirs(model_dir, exist_ok=True)

    model_name = hparams.get("name", "mace_finetuned")

    config = {
        "name":           model_name,
        "train_file":     dataset_info["train_file"],
        "valid_file":     dataset_info["val_file"],
        "model":          "MACE",
        "r_max":          hparams.get("r_max", 5.0),
        "num_channels":   hparams.get("num_channels", 128),
        "max_L":          hparams.get("max_L", 1),
        "correlation":    hparams.get("correlation", 3),
        "max_num_epochs": hparams.get("max_num_epochs", 200),
        "batch_size":     hparams.get("batch_size", 4),
        "lr":             hparams.get("learning_rate", 0.01),
        "energy_weight":  hparams.get("energy_weight", 1.0),
        "forces_weight":  hparams.get("forces_weight", 100.0),
        "ema":            True,
        "ema_decay":      0.99,
        "amsgrad":        True,
        "default_dtype":  "float64",
        "device":         hparams.get("device", "cuda"),
        "save_cpu":       True,
        "results_dir":    model_dir,
        "log_dir":        os.path.join(model_dir, "logs"),
    }

    if foundation_model:
        config["foundation_model"] = foundation_model

    # Write config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Build CLI
    cli = ["mace_run_train"]
    for k, v in config.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        else:
            cli += [f"--{k}", str(v)]

    stdout_log = os.path.join(config["log_dir"], "stdout.log")
    stderr_log = os.path.join(config["log_dir"], "stderr.log")
    os.makedirs(config["log_dir"], exist_ok=True)

    logger.info(f"Starting MACE training ({model_name})...")

    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(
            cli, stdout=out, stderr=err,
            cwd=working_dir,
            timeout=int(timeout_hours * 3600),
        )

    if proc.returncode != 0:
        with open(stderr_log) as f:
            tail = "".join(f.readlines()[-40:])
        raise RuntimeError(f"MACE training failed (exit {proc.returncode}):\n{tail}")

    model_file = os.path.join(model_dir, f"{model_name}.model")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found after training: {model_file}")

    return {"model_file": model_file, "config": config, "status": "success"}


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION  (backend dispatch)
# ═══════════════════════════════════════════════════════════════════

def validate_model(
    backend: str,
    model_file: str,
    val_file: str,
    working_dir: str,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """
    Compute energy/force MAE on held-out validation data.

    Returns:
        {
            "energy_mae_meV": float,
            "force_mae_meV_A": float,
            "max_force_error_meV_A": float,
            "n_evaluated": int,
            "passed": bool,
        }
    """
    if backend == "mace":
        return _mace_validate(model_file, val_file, working_dir, n_samples)
    else:
        raise NotImplementedError(
            f"Validation for {backend} not yet implemented."
        )


def _mace_validate(
    model_file: str, val_file: str,
    working_dir: str, n_samples: int,
) -> Dict[str, Any]:
    """Validate a MACE model on extXYZ reference data."""
    import ase.io
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=model_file, device="cpu", default_dtype="float64"
    )
    frames = list(ase.io.read(val_file, index=":"))[:n_samples]

    e_err, f_err = [], []

    for frame in frames:
        ref_e = frame.calc.results["energy"] / len(frame)
        ref_f = frame.calc.results["forces"]

        test = frame.copy()
        test.calc = calc
        pred_e = test.get_potential_energy() / len(test)
        pred_f = test.get_forces()

        e_err.append(abs(pred_e - ref_e) * 1000)
        f_err.extend(np.linalg.norm(pred_f - ref_f, axis=1) * 1000)

    energy_mae = float(np.mean(e_err))
    force_mae = float(np.mean(f_err))
    max_f_err = float(np.max(f_err))

    passed = energy_mae < 5.0 and force_mae < 100.0

    return {
        "energy_mae_meV": energy_mae,
        "force_mae_meV_A": force_mae,
        "max_force_error_meV_A": max_f_err,
        "n_evaluated": len(frames),
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════
#  DFT INPUT GENERATION
# ═══════════════════════════════════════════════════════════════════

def extract_problematic_frames(
    trajectory_file: str,
    model_file: str,
    backend: str = "mace",
    max_frames: int = 500,
    top_n: int = 20,
    device: str = "cpu",
) -> List[Any]:
    """
    Read a trajectory, score each frame by uncertainty, return the
    top-N most uncertain frames as ase.Atoms objects.

    These are the frames that should be sent to DFT for active learning.
    """
    import ase.io

    frames = list(ase.io.read(trajectory_file, index=":"))
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step]

    unc = evaluate_uncertainty(backend, model_file, frames, device)

    # Sort by uncertainty descending
    scored = sorted(
        zip(unc["per_structure"], frames),
        key=lambda x: x[0].get("max_force_uncertainty", 0),
        reverse=True,
    )

    selected = [frame for _, frame in scored[:top_n]]
    logger.info(
        f"Selected {len(selected)} high-uncertainty frames "
        f"from {len(frames)} trajectory frames"
    )
    return selected


def write_dft_inputs(
    structures: List[Any],
    working_dir: str,
    dft_code: str = "vasp",
    dft_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write DFT input files for a list of structures.

    Supports VASP, CP2K, and a generic extXYZ dump for other codes.

    Returns:
        {
            "dft_code": str,
            "directories": [str],    # one per structure
            "n_structures": int,
            "instructions": str,     # human-readable next steps
        }
    """
    import ase.io

    os.makedirs(working_dir, exist_ok=True)
    settings = dft_settings or {}
    directories = []

    for i, atoms in enumerate(structures):
        calc_dir = os.path.join(working_dir, f"frame_{i:04d}")
        os.makedirs(calc_dir, exist_ok=True)

        if dft_code == "vasp":
            _write_vasp_inputs(atoms, calc_dir, settings)
        elif dft_code == "cp2k":
            _write_cp2k_inputs(atoms, calc_dir, settings)
        else:
            # Generic: just write the structure
            ase.io.write(
                os.path.join(calc_dir, "structure.xyz"),
                atoms, format="extxyz",
            )

        directories.append(calc_dir)

    instructions = {
        "vasp": (
            f"Run VASP in each of the {len(directories)} directories. "
            f"After completion, collect OUTCAR files and convert to extXYZ "
            f"with: ase convert */OUTCAR collected_dft.xyz"
        ),
        "cp2k": (
            f"Run CP2K in each of the {len(directories)} directories. "
            f"After completion, parse forces from output files."
        ),
    }.get(dft_code, (
        f"Run your DFT code on the {len(directories)} structures in "
        f"{working_dir}/frame_XXXX/structure.xyz. "
        f"Collect results into a single extXYZ file with energy and forces."
    ))

    return {
        "dft_code": dft_code,
        "directories": directories,
        "n_structures": len(structures),
        "instructions": instructions,
    }


def _write_vasp_inputs(atoms, calc_dir, settings):
    """Write POSCAR + template INCAR + KPOINTS for a single structure."""
    import ase.io

    ase.io.write(os.path.join(calc_dir, "POSCAR"), atoms, format="vasp")

    encut = settings.get("encut", 520)
    kpoints = settings.get("kpoints", [3, 3, 3])

    incar = f"""SYSTEM = MLIP active learning
ENCUT = {encut}
PREC = Accurate
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
IBRION = -1
NSW = 0
LREAL = Auto
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
    with open(os.path.join(calc_dir, "INCAR"), "w") as f:
        f.write(incar)

    with open(os.path.join(calc_dir, "KPOINTS"), "w") as f:
        f.write(f"Automatic\n0\nGamma\n{kpoints[0]} {kpoints[1]} {kpoints[2]}\n0 0 0\n")


def _write_cp2k_inputs(atoms, calc_dir, settings):
    """Write structure + template CP2K input."""
    import ase.io

    ase.io.write(os.path.join(calc_dir, "structure.xyz"), atoms, format="xyz")
    # Minimal template — user should customize
    with open(os.path.join(calc_dir, "cp2k.inp"), "w") as f:
        f.write("# CP2K input template — customize for your system\n")
        f.write("# Structure: structure.xyz\n")


# ═══════════════════════════════════════════════════════════════════
#  LAMMPS INPUT GENERATION  (backend dispatch)
# ═══════════════════════════════════════════════════════════════════

def generate_lammps_input(
    backend: str,
    model_file: str,
    elements: List[str],
    working_dir: str,
    timestep: float = 0.5,
    temperature: float = 300.0,
    pressure: Optional[float] = None,
) -> str:
    """
    Generate LAMMPS input file for the given backend.
    Returns path to the written input file.
    """
    os.makedirs(working_dir, exist_ok=True)

    if backend == "mace":
        return _mace_lammps_input(model_file, elements, working_dir,
                                   timestep, temperature, pressure)
    elif backend == "nequip":
        return _nequip_lammps_input(model_file, elements, working_dir,
                                     timestep, temperature, pressure)
    elif backend == "deepmd":
        return _deepmd_lammps_input(model_file, elements, working_dir,
                                     timestep, temperature, pressure)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _mace_lammps_input(model_file, elements, working_dir,
                        timestep, temperature, pressure):
    el_str = " ".join(elements)
    ensemble_fix = (
        f"fix 1 all npt temp {temperature} {temperature} 0.1 "
        f"iso {pressure} {pressure} 1.0"
        if pressure is not None
        else f"fix 1 all nvt temp {temperature} {temperature} 0.1"
    )
    content = f"""# LAMMPS input — MACE potential
# Model: {os.path.basename(model_file)}
# Elements: {el_str}

units          metal
atom_style     atomic
boundary       p p p

read_data      system.data

pair_style     mace no_domain_decomposition
pair_coeff     * * {model_file} {el_str}

neighbor       2.0 bin
neigh_modify   every 1 delay 0 check yes

thermo         100
thermo_style   custom step temp press pe ke etotal vol density

dump           traj all custom 1000 traj.lammpstrj id type x y z fx fy fz

min_style      cg
minimize       1.0e-6 1.0e-8 1000 10000

velocity       all create {temperature} 12345 mom yes rot yes
timestep       {timestep}e-3

{ensemble_fix}
run            100000
"""
    path = os.path.join(working_dir, "in.lammps")
    with open(path, "w") as f:
        f.write(content)
    return path


def _nequip_lammps_input(model_file, elements, working_dir,
                          timestep, temperature, pressure):
    el_str = " ".join(elements)
    ensemble_fix = (
        f"fix 1 all npt temp {temperature} {temperature} 0.1 "
        f"iso {pressure} {pressure} 1.0"
        if pressure is not None
        else f"fix 1 all nvt temp {temperature} {temperature} 0.1"
    )
    content = f"""# LAMMPS input — NequIP potential
units          metal
atom_style     atomic
boundary       p p p

read_data      system.data

pair_style     nequip
pair_coeff     * * {model_file} {el_str}

neighbor       2.0 bin
neigh_modify   every 1 delay 0 check yes

thermo         100
thermo_style   custom step temp press pe ke etotal vol density
dump           traj all custom 1000 traj.lammpstrj id type x y z fx fy fz

velocity       all create {temperature} 12345 mom yes rot yes
timestep       {timestep}e-3

{ensemble_fix}
run            100000
"""
    path = os.path.join(working_dir, "in.lammps")
    with open(path, "w") as f:
        f.write(content)
    return path


def _deepmd_lammps_input(model_file, elements, working_dir,
                          timestep, temperature, pressure):
    el_str = " ".join(elements)
    ensemble_fix = (
        f"fix 1 all npt temp {temperature} {temperature} 0.1 "
        f"iso {pressure} {pressure} 1.0"
        if pressure is not None
        else f"fix 1 all nvt temp {temperature} {temperature} 0.1"
    )
    content = f"""# LAMMPS input — DeePMD potential
units          metal
atom_style     atomic
boundary       p p p

read_data      system.data

pair_style     deepmd {model_file}
pair_coeff     * * {el_str}

neighbor       2.0 bin
neigh_modify   every 1 delay 0 check yes

thermo         100
thermo_style   custom step temp press pe ke etotal vol density
dump           traj all custom 1000 traj.lammpstrj id type x y z fx fy fz

velocity       all create {temperature} 12345 mom yes rot yes
timestep       {timestep}e-3

{ensemble_fix}
run            100000
"""
    path = os.path.join(working_dir, "in.lammps")
    with open(path, "w") as f:
        f.write(content)
    return path
