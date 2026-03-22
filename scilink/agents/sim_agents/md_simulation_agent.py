# agents/sim_agents/md_simulation_agent.py
"""
MD simulation agent. Handles classical atomistic dynamics
across engines (LAMMPS, GROMACS, OpenMM, etc.) via skills.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base_simulation_agent import SimulationAgent

_TOOL_REGISTRY: Dict[str, Any] = {}
try:
    from ...tools import lammps_tools
    _TOOL_REGISTRY["lammps"] = lammps_tools
except ImportError:
    pass


class MDSimulationAgent(SimulationAgent):
    """
    MD-specific simulation agent.

    Adds MD concepts: ensemble, temperature, pressure, timestep,
    equilibration/production phases, force field integration, staged runs.

    The base class provides: skill loading, LLM helpers, validation,
    refinement, output cleaning.
    """

    SKILL_DOMAIN = "molecular_dynamics"
    EXTENSION_MAP = {
        "lammps": [".data", ".lmp"],
        "gromacs": [".gro", ".g96"],
        "openmm":  [".pdb", ".cif"],
    }
    TOOL_REGISTRY = _TOOL_REGISTRY

    def __init__(self, working_dir: str, **kwargs):
        super().__init__(working_dir=working_dir, **kwargs)
        self._last_plan: Optional[Dict[str, Any]] = None
        self._last_system_info: Optional[Dict[str, Any]] = None

    # ================================================================
    # SYSTEM ANALYSIS
    # ================================================================

    def analyze_system(self, structure_file: str) -> Dict[str, Any]:
        self.logger.info(f"Analyzing: {structure_file}")

        if self.tools_module and hasattr(self.tools_module, "parse_data_file"):
            try:
                info = self.tools_module.parse_data_file(structure_file)
                if info.get("atom_count", 0) > 0:
                    self.logger.info(
                        f"Analysis (tools): {info['atom_count']} atoms, "
                        f"{info.get('system_category', 'unknown')}"
                    )
                    return info
            except Exception as e:
                self.logger.warning(f"Tool analysis failed: {e}")

        try:
            from ase.io.lammpsdata import read_lammps_data
            atoms = read_lammps_data(structure_file, style="full", units="real")
            ec = {}
            for s in atoms.get_chemical_symbols():
                ec[s] = ec.get(s, 0) + 1
            return {
                "atom_count": len(atoms),
                "elements": sorted(ec.keys()),
                "element_counts": ec,
                "box_dimensions": atoms.get_cell().diagonal().tolist(),
                "has_water": (
                    "O" in ec and "H" in ec
                    and ec.get("H", 0) >= 2 * ec.get("O", 0)
                ),
                "has_ions": any(
                    x in ec for x in ["Na", "Cl", "K", "Ca", "Mg"]
                ),
                "has_organic": "C" in ec,
                "system_category": "unknown",
            }
        except Exception:
            pass

        return self._llm_analyze_structure(structure_file)

    def _llm_analyze_structure(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            header = f.read(5000)
        ctx = self._get_skill_context(section="analysis")
        prompt = (
            "Analyze this structure file.\n\n"
            f"{ctx}\n\n"
            "FILE (first 5000 chars):\n"
            f"{header}\n\n"
            "Return JSON:\n"
            '{"atom_count": int, "elements": [...], "element_counts": {},'
            ' "box_dimensions": [x,y,z], "has_water": bool, "has_ions": bool,'
            ' "has_organic": bool, "has_metal": bool, "system_category": str}'
        )
        try:
            return self._generate_json(prompt)
        except Exception:
            return {
                "atom_count": 0,
                "elements": [],
                "element_counts": {},
                "system_category": "unknown",
            }

    # ================================================================
    # PLANNING
    # ================================================================

    def plan_simulation(
        self,
        research_goal: str,
        system_info: Dict[str, Any],
        temperature: float = 300.0,
        pressure: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        self.logger.info(f"Planning: {research_goal}")

        elements_str = ", ".join(
            f"{e}: {c}"
            for e, c in system_info.get("element_counts", {}).items()
        )
        planning = self._get_skill_context(section="planning")

        prompt = (
            "Recommend MD simulation parameters for this research goal.\n\n"
            f'GOAL: "{research_goal}"\n\n'
            "SYSTEM:\n"
            f"- Elements: {elements_str}\n"
            f"- Atoms: {system_info.get('atom_count', 0)}\n"
            f"- Category: {system_info.get('system_category', 'unknown')}\n"
            f"- Water: {'Yes' if system_info.get('has_water') else 'No'}\n"
            f"- Ions: {'Yes' if system_info.get('has_ions') else 'No'}\n"
            f"- Metal: {'Yes' if system_info.get('has_metal') else 'No'}\n"
            f"- Has bonds: {'Yes' if system_info.get('has_bonds') else 'No'}\n"
            f"- Has vacuum: {'Yes' if system_info.get('has_vacuum') else 'No'}\n\n"
            f"{planning}\n\n"
            "Use the tables above to select the correct unit system, timestep, and\n"
            "damping constants for this system type.\n\n"
            "Return JSON:\n"
            "{\n"
            '    "simulation_technique": "standard_md",\n'
            '    "ensemble": "NPT",\n'
            '    "temperature": 300.0,\n'
            '    "pressure": 1.0,\n'
            '    "timestep": 2.0,\n'
            '    "equilibration_time": 0.5,\n'
            '    "production_time": 1.5,\n'
            '    "requires_multiple_simulations": false,\n'
            '    "number_of_simulations": 1,\n'
            '    "required_outputs": ["energy", "trajectory"],\n'
            '    "methodology_description": "brief explanation"\n'
            "}"
        )
        try:
            params = self._generate_json(prompt)
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            params = {}

        params.setdefault("simulation_technique", "standard_md")
        params.setdefault("ensemble", "NPT")
        params.setdefault("temperature", temperature)
        params.setdefault("pressure", pressure)
        params.setdefault("timestep", 2.0)
        params.setdefault("equilibration_time", 0.5)
        params.setdefault("production_time", 1.5)
        params.setdefault("requires_multiple_simulations", False)
        params.setdefault("number_of_simulations", 1)
        params.setdefault("required_outputs", ["energy", "trajectory"])

        for k, v in kwargs.items():
            params[k] = v

        return params

    # ================================================================
    # GENERATION
    # ================================================================

    def generate_simulation(
        self,
        structure_file: str,
        research_goal: str,
        system_description: Optional[str] = None,
        temperature: float = 300.0,
        pressure: float = 1.0,
        force_field_files: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._auto_select_skill(structure_file)

        system_info = self.analyze_system(structure_file)
        self._last_system_info = system_info

        if not system_description:
            system_description = self._describe_system(system_info)

        plan = self.plan_simulation(
            research_goal=research_goal,
            system_info=system_info,
            temperature=temperature,
            pressure=pressure,
            **kwargs,
        )
        self._last_plan = plan

        script = self._generate_md_input(
            structure_file,
            research_goal,
            system_description,
            system_info,
            plan,
        )

        if force_field_files:
            script = self._integrate_force_fields(script, force_field_files)

        script = self._clean_and_fix(script, plan)

        script_path = self.working_dir / "run.lammps"
        script_path.write_text(script)

        validation = self._validate(str(script_path), system_info, plan)

        if not validation.get("valid", True) and validation.get("errors"):
            self.logger.warning("Validation failed, fixing...")
            script = self._attempt_fix(script, validation["errors"], plan)
            script_path.write_text(script)
            validation = self._validate(str(script_path), system_info, plan)

        readme = self._generate_readme(
            research_goal,
            system_description,
            system_info,
            plan,
            str(script_path),
        )

        return {
            "script_path": str(script_path),
            "readme_path": readme,
            "data_path": structure_file,
            "system_info": system_info,
            "simulation_parameters": plan,
            "validation": validation,
            "skill_used": self.skill_name,
        }

    def _generate_md_input(
        self, structure_file, goal, desc, info, plan
    ) -> str:
        data_filename = os.path.basename(structure_file)

        type_info = ""
        if self.tools_module and hasattr(self.tools_module, "format_type_info"):
            try:
                type_info = self.tools_module.format_type_info(structure_file)
            except Exception:
                pass

        ts = plan.get("timestep", 2.0)
        equil_steps = int(
            (plan.get("equilibration_time", 0.5) * 1e6) / ts
        )
        prod_steps = int(
            (plan.get("production_time", 1.5) * 1e6) / ts
        )

        elements_str = ", ".join(
            f"{e}: {c}"
            for e, c in info.get("element_counts", {}).items()
        )

        implementation = self._get_skill_context(section="implementation")
        planning = self._get_skill_context(section="planning")
        validation_rules = self._get_skill_context(section="validation")
        has_skill = bool(implementation)

        multi_block = ""
        if (
            plan.get("requires_multiple_simulations")
            and plan.get("number_of_simulations", 1) > 1
        ):
            multi_block = self._build_multi_sim_block(plan)

        if has_skill:
            prompt = (
                "You are an expert molecular dynamics simulation engineer.\n"
                "Generate a complete, runnable input file.\n\n"
                "## Goal\n"
                f"{goal}\n\n"
                "## System\n"
                f"{desc}\n"
                f"- Elements: {elements_str}\n"
                f"- Atoms: {info.get('atom_count', 0)}\n"
                f"- Category: {info.get('system_category', 'unknown')}\n"
                f"- Box: {info.get('box_dimensions', [])}\n"
                f"- Vacuum: {'Yes -- slab' if info.get('has_vacuum') else 'No'}\n\n"
                f"{type_info}\n\n"
                "## Plan\n"
                f"- Technique: {plan.get('simulation_technique')}\n"
                f"- Ensemble: {plan.get('ensemble')}\n"
                f"- Temperature: {plan.get('temperature')} K\n"
                f"- Pressure: {plan.get('pressure')} atm\n"
                f"- Timestep: {ts}\n"
                f"- Equilibration: {plan.get('equilibration_time')} ns ({equil_steps} steps)\n"
                f"- Production: {plan.get('production_time')} ns ({prod_steps} steps)\n"
                f"- Outputs: {plan.get('required_outputs', [])}\n"
                f"{multi_block}\n\n"
                "## Implementation Templates\n"
                f"{implementation}\n\n"
                "## Parameter Guidelines\n"
                f"{planning}\n\n"
                "## Validation Rules (must satisfy)\n"
                f"{validation_rules}\n\n"
                "RULES:\n"
                "1. Match unit system and pair_style to system category per the tables.\n"
                "2. Use literal values -- no unresolved template variables.\n"
                f"3. Structure file: {data_filename}\n"
                "4. If data file has embedded Pair Coeffs, do not add pair_coeff commands.\n"
                "5. For external potentials, add pair_coeff referencing the file with a comment.\n\n"
                "Return ONLY the input file. No markdown."
            )
        else:
            prompt = (
                "You are an MD simulation expert. No engine-specific skill loaded.\n\n"
                f"GOAL: {goal}\n"
                f"SYSTEM: {desc} ({info.get('atom_count', 0)} atoms)\n"
                f"PLAN: {plan.get('ensemble')} at {plan.get('temperature')} K, dt={ts},\n"
                f"      equil {equil_steps} steps, prod {prod_steps} steps\n"
                f"FILE: {data_filename}\n\n"
                "# GENERATED WITHOUT SKILL -- VERIFY BEFORE RUNNING\n\n"
                "Return ONLY the input file. No markdown."
            )

        return self._generate_text(prompt)

    def _build_multi_sim_block(self, plan):
        n = plan.get("number_of_simulations", 1)
        tech = plan.get("simulation_technique", "")
        var = plan.get("variable_parameter", "")
        vals = plan.get("variable_values", [])
        vals_str = ", ".join(str(v) for v in vals[:5])
        if len(vals) > 5:
            vals_str += f"... ({len(vals)} total)"
        return (
            "\n## Multi-Simulation\n"
            f"- Technique: {tech}\n"
            f"- Runs: {n}\n"
            f"- Variable: {var}\n"
            f"- Values: {vals_str}\n"
        )

    # ================================================================
    # FORCE FIELD INTEGRATION
    # ================================================================

    def _integrate_force_fields(self, script, ff_files):
        if self.tools_module and hasattr(
            self.tools_module, "integrate_force_field_files"
        ):
            return self.tools_module.integrate_force_field_files(
                script, ff_files, str(self.working_dir)
            )
        self.logger.warning("No FF integration tool")
        header = "\n".join(
            f"# FF: {n} = {p}" for n, p in ff_files.items()
        )
        return header + "\n\n" + script

    # ================================================================
    # CLEANING
    # ================================================================

    def _clean_and_fix(self, script, plan):
        script = self._clean_output(script)
        if self.tools_module and hasattr(
            self.tools_module, "substitute_variables"
        ):
            script = self.tools_module.substitute_variables(
                script,
                temperature=plan.get("temperature", 300.0),
                pressure=plan.get("pressure", 1.0),
                timestep=plan.get("timestep", 2.0),
            )
        return script

    # ================================================================
    # STAGED SIMULATION
    # ================================================================

    def generate_staged_simulation(
        self, structure_file, research_goal, **kw
    ):
        result = self.generate_simulation(
            structure_file=structure_file,
            research_goal=research_goal,
            **kw,
        )
        full_script = Path(result["script_path"]).read_text()
        plan = result["simulation_parameters"]
        impl = self._get_skill_context(section="implementation")

        prompt = (
            "Split this simulation into 2-4 checkpointed stages.\n\n"
            "SCRIPT:\n"
            f"{full_script}\n\n"
            f"{impl}\n\n"
            "Each stage: complete, standalone, runnable. First reads data file,\n"
            "later stages read restart. All include force field commands.\n"
            "Use literal values. Write restart at end of each stage.\n\n"
            'Return JSON: {"equilibration": "script...", "production": "script..."}'
        )
        try:
            stages = self._generate_json(prompt)
        except Exception:
            stages = {"production": full_script}

        stage_scripts = {}
        for name, content in stages.items():
            if not isinstance(content, str):
                continue
            content = self._clean_and_fix(content, plan)
            path = self.working_dir / f"run_{name}.lammps"
            path.write_text(content)
            stage_scripts[name] = str(path)

        result.update(
            staged_scripts=stage_scripts,
            stages=list(stage_scripts.keys()),
            is_staged=True,
        )
        return result

    # ================================================================
    # SYSTEM DESCRIPTION
    # ================================================================

    def _describe_system(self, info):
        cat = info.get("system_category", "unknown")
        parts = []
        non_molecular = {"O", "H", "C", "N", "S", "P", "F", "Cl"}

        if cat == "metal":
            metals = [
                e for e in info.get("elements", [])
                if e not in non_molecular
            ]
            parts.append(" ".join(metals) + " metal")
        elif cat == "semiconductor":
            parts.append(
                " ".join(info.get("elements", [])) + " semiconductor"
            )
        elif cat == "oxide":
            parts.append("metal oxide")
        elif cat == "ionic":
            parts.append("ionic crystal")
        else:
            if info.get("has_water"):
                parts.append("water")
            if info.get("has_ions"):
                ions = [
                    e for e in info.get("elements", [])
                    if e in ["Na", "K", "Cl", "Ca", "Mg"]
                ]
                if ions:
                    parts.append("+".join(ions) + " ions")
                else:
                    parts.append("ions")
            if info.get("has_organic"):
                parts.append("organic molecules")

        if not parts:
            parts.append("molecular system")

        desc = " with ".join(parts)

        if info.get("has_vacuum"):
            axis = info.get("vacuum_axis", "z")
            desc += f" (slab, vacuum {axis})"

        return f"{desc} ({info.get('atom_count', 0)} atoms)"

    # ================================================================
    # README
    # ================================================================

    def _generate_readme(self, goal, desc, info, plan, script_path):
        p = self.working_dir / "README.md"
        with open(p, "w") as f:
            f.write(f"# MD Simulation: {desc}\n\n")
            f.write(f"**Skill**: {self.skill_name or 'none'}\n\n")
            f.write(f"## Goal\n{goal}\n\n")
            f.write("## System\n")
            for el, c in info.get("element_counts", {}).items():
                f.write(f"- {el}: {c}\n")
            f.write(
                f"- Category: {info.get('system_category', 'unknown')}\n\n"
            )
            f.write("## Parameters\n")
            param_keys = [
                "ensemble",
                "temperature",
                "pressure",
                "timestep",
                "equilibration_time",
                "production_time",
            ]
            for k in param_keys:
                f.write(f"- {k}: {plan.get(k)}\n")
            f.write(f"\n## Run\n")
            f.write(f"cd {self.working_dir}\n")
            f.write(f"lmp -in {os.path.basename(script_path)}\n")
        return str(p)
