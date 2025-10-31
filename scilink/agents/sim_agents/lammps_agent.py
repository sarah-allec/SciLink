import os
import re
import shutil
import logging
import json
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from ase import io
from ase.io.lammpsdata import read_lammps_data
from .instruct import LAMMPS_INPUT_GENERATION_TEMPLATE

class LAMMPSSimulationAgent:
    def __init__(self, working_dir: str, api_key: Optional[str] = None):
        """
        Initialize the LAMMPS simulation agent.
        
        Args:
            working_dir: Directory for output files
            api_key: API key for Google Gemini API (optional if set in environment)
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _integrate_force_field_files(self, script_text: str, force_field_files: Dict[str, str]) -> str:
        """
        Integrate force field parameter files into the LAMMPS script.
        
        Args:
            script_text: Original LAMMPS script text
            force_field_files: Dictionary with paths to force field parameter files
        
        Returns:
            Updated LAMMPS script with force field parameters
        """
        if not force_field_files:
            return script_text
            
        lines = script_text.split('\n')
        
        # Find the position to insert force field parameters
        # Usually after reading the data file
        insert_pos = 0
        for i, line in enumerate(lines):
            if "read_data" in line:
                insert_pos = i + 1
                break
                
        # If no insert position found, insert at the beginning
        if insert_pos == 0:
            for i, line in enumerate(lines):
                if not line.startswith("#") and line.strip():
                    insert_pos = i
                    break
                    
        # Create the insertion text
        insertion = [
            "",
            "# Include force field parameters",
        ]
        
        # Add main parameter file
        if "main" in force_field_files:
            # Copy the file to the working directory
            main_file = os.path.basename(force_field_files["main"])
            shutil.copy2(force_field_files["main"], os.path.join(self.working_dir, main_file))
            insertion.append(f"include {main_file}")
            
        # Add additional parameter files
        for name, path in force_field_files.get("additional", {}).items():
            if os.path.exists(path):
                # Copy the file to the working directory
                add_file = f"{name}.lammps"
                shutil.copy2(path, os.path.join(self.working_dir, add_file))
                insertion.append(f"include {add_file}")
                
        insertion.append("")
        
        # Insert into the script
        updated_lines = lines[:insert_pos] + insertion + lines[insert_pos:]
        return '\n'.join(updated_lines)
    
    def generate_simulation(self, 
                          data_file: str, 
                          research_goal: str,
                          system_description: Optional[str] = None,
                          temperature: float = 300.0,
                          pressure: float = 1.0,
                          force_field_files: Optional[Dict[str, str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate LAMMPS simulation(s) based on a research goal.
        
        Args:
            data_file: Path to LAMMPS data file
            research_goal: Research objective in natural language
            system_description: Description of the molecular system (optional)
            temperature: Default temperature in K
            pressure: Default pressure in atm
            force_field_files: Dictionary with paths to force field parameter files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generated simulation info
        """
        # Copy the data file to the working directory
        local_data_file = self._copy_data_file(data_file)
        
        # Analyze the system
        system_info = self.analyze_system(data_file)
        
        # If system description is not provided, generate one based on analysis
        if not system_description:
            system_description = self._generate_system_description(system_info)
        
        # Determine simulation parameters based on research goal
        simulation_params = self._determine_simulation_parameters(
            research_goal=research_goal,
            system_info=system_info,
            temperature=temperature,
            pressure=pressure,
            **kwargs
        )
        
        # Generate LAMMPS script
        script_text = self._generate_script(
            data_filename=os.path.basename(local_data_file),
            research_goal=research_goal,
            system_description=system_description,
            system_info=system_info,
            **simulation_params
        )
        
        # Add force field parameters - either from files or generated
        if force_field_files:
            # Use provided force field files
            self.logger.info("Integrating provided force field files")
            script_text = self._integrate_force_field_files(script_text, force_field_files)
        else:
            # Generate basic force field parameters
            self.logger.info("Generating basic force field parameters")
            script_text = self._ensure_force_field_parameters(script_text, system_info)
        
        # Save the script
        script_path = os.path.join(self.working_dir, "run.lammps")
        with open(script_path, 'w') as f:
            f.write(script_text)
        
        # Create a README with analysis instructions
        readme_path = self._generate_readme(
            research_goal=research_goal,
            system_description=system_description,
            system_info=system_info,
            simulation_params=simulation_params,
            script_path=script_path
        )
        
        return {
            "script_path": script_path,
            "readme_path": readme_path,
            "data_path": local_data_file,
            "system_info": system_info,
            "simulation_parameters": simulation_params
        }    
    def analyze_system(self, data_file: str) -> Dict[str, Any]:
        """Analyze a LAMMPS data file using ASE to identify its components."""
        self.logger.info(f"Analyzing system from {data_file}")
        try:
            # Read LAMMPS data file
            atoms = read_lammps_data(data_file, style="full", units="real")
            
            # Extract basic information
            element_counts = {}
            for symbol in atoms.get_chemical_symbols():
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
                
            # Identify common components
            has_water = ('O' in element_counts and 'H' in element_counts and 
                        element_counts.get('H', 0) >= 2 * element_counts.get('O', 0))
            has_ions = any(x in element_counts for x in ['Na', 'Cl', 'K', 'Ca', 'Mg'])
            has_organic = 'C' in element_counts
            
            # Get bond and angle information from the data file
            bond_types, angle_types = self._extract_bond_angle_types(data_file)
            
            system_info = {
                "atom_count": len(atoms),
                "elements": list(element_counts.keys()),
                "element_counts": element_counts,
                "box_dimensions": atoms.get_cell().diagonal().tolist(),
                "has_water": has_water,
                "has_ions": has_ions,
                "has_organic": has_organic,
                "bond_types": bond_types,
                "angle_types": angle_types
            }
            
            self.logger.info(f"System analysis complete: {system_info}")
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing data file: {e}")
            # Fallback to minimal system info
            return {
                "atom_count": 0,
                "elements": [],
                "element_counts": {},
                "has_water": False,
                "has_ions": False,
                "has_organic": False,
                "bond_types": 0,
                "angle_types": 0
            }
    
    def _extract_bond_angle_types(self, data_file: str) -> tuple:
        """Extract bond and angle type information from a LAMMPS data file."""
        bond_types = 0
        angle_types = 0
        
        try:
            with open(data_file, 'r') as f:
                content = f.read()
                
                # Extract bond types
                match = re.search(r'(\d+)\s+bond\s+types', content)
                if match:
                    bond_types = int(match.group(1))
                    
                # Extract angle types
                match = re.search(r'(\d+)\s+angle\s+types', content)
                if match:
                    angle_types = int(match.group(1))
                    
        except Exception as e:
            self.logger.error(f"Error extracting bond/angle types: {e}")
            
        return bond_types, angle_types
    
    def _copy_data_file(self, source_path: str) -> str:
        """Copy the data file to the working directory."""
        # Default to "system.data" as the destination filename
        dest_filename = "system.data"
        dest_path = os.path.join(self.working_dir, dest_filename)
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        self.logger.info(f"Copied data file from {source_path} to {dest_path}")
        
        return dest_path
    
    def _generate_system_description(self, system_info: Dict[str, Any]) -> str:
        """Generate a system description based on analysis."""
        elements = system_info.get("elements", [])
        atom_count = system_info.get("atom_count", 0)
        
        description_parts = []
        
        if system_info.get("has_water", False):
            description_parts.append("water")
        
        if system_info.get("has_ions", False):
            ions = [e for e in elements if e in ["Na", "K", "Cl", "Ca", "Mg"]]
            if ions:
                description_parts.append("+".join(ions) + " ions")
            else:
                description_parts.append("ions")
        
        if system_info.get("has_organic", False) and "C" in elements:
            description_parts.append("organic molecules")
        
        if not description_parts:
            # Generic description
            description_parts.append("molecular system")
        
        description = " with ".join(description_parts)
        return f"{description} ({atom_count} atoms)"
    
    def _determine_simulation_parameters(self, 
                                       research_goal: str, 
                                       system_info: Dict[str, Any],
                                       temperature: float = 300.0,
                                       pressure: float = 1.0,
                                       **kwargs) -> Dict[str, Any]:
        """
        Determine simulation parameters based on the research goal and system info.
        
        This function uses the LLM to analyze the research goal and recommend simulation settings.
        """
        self.logger.info(f"Determining simulation parameters for research goal: {research_goal}")
        
        # Construct element counts string
        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("element_counts", {}).items()])
        
        # Prompt for the LLM to analyze the research goal and determine appropriate simulation parameters
        prompt = f"""
        Analyze the following research goal for a molecular dynamics simulation and recommend appropriate parameters.
        
        RESEARCH GOAL: "{research_goal}"
        
        SYSTEM INFORMATION:
        - Elements: {elements_str}
        - Total atoms: {system_info.get('atom_count', 0)}
        - Contains water: {'Yes' if system_info.get('has_water', False) else 'No'}
        - Contains ions: {'Yes' if system_info.get('has_ions', False) else 'No'}
        - Contains organic molecules: {'Yes' if system_info.get('has_organic', False) else 'No'}
        
        Based on this research goal and system information, provide the following:
        1. What properties need to be calculated
        2. What ensemble is most appropriate (NVE, NVT, NPT)
        3. What simulation time is needed (in nanoseconds)
        4. What temperature(s) to use (in K)
        5. What pressure to use (in atm), if applicable
        6. What timestep to use (in fs)
        7. What specific analysis commands are needed in LAMMPS
        
        Respond with a JSON object with the following structure:
        ```json
        {{
            "properties_to_calculate": ["list of properties"],
            "ensemble": "NPT",
            "simulation_time": 2.0,
            "temperature": 300.0,
            "pressure": 1.0,
            "timestep": 2.0,
            "equilibration_time": 0.5,
            "production_time": 1.5,
            "required_outputs": ["density", "rdf", "msd", etc.],
            "special_settings": {{
                "additional parameters as key-value pairs"
            }}
        }}
        ```
        
        Include only the JSON response with no additional text.
        """
        
        try:
            # Generate response from LLM
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            params = json.loads(response.text)
            
            # Add default values if missing
            params.setdefault("ensemble", "NPT")
            params.setdefault("simulation_time", 2.0)
            params.setdefault("temperature", temperature)
            params.setdefault("pressure", pressure)
            params.setdefault("timestep", 2.0)
            params.setdefault("equilibration_time", params["simulation_time"] * 0.25)
            params.setdefault("production_time", params["simulation_time"] * 0.75)
            
            # Override with explicit kwargs if provided
            for key, value in kwargs.items():
                params[key] = value
                
            self.logger.info(f"Determined simulation parameters: {params}")
            return params
            
        except Exception as e:
            self.logger.error(f"Error determining simulation parameters: {e}")
            # Fallback to default parameters
            default_params = {
                "properties_to_calculate": ["energy", "structure"],
                "ensemble": "NPT",
                "simulation_time": 2.0,
                "temperature": temperature,
                "pressure": pressure,
                "timestep": 2.0,
                "equilibration_time": 0.5,
                "production_time": 1.5,
                "required_outputs": ["energy", "trajectory"]
            }
            return default_params
    
    def _generate_script(self, **kwargs) -> str:
        """
        Generate a LAMMPS script based on the research goal and system information.
        
        This function uses the LLM to generate a complete LAMMPS input script.
        """
        system_info = kwargs.get("system_info", {})
        data_filename = kwargs.get("data_filename", "system.data")
        research_goal = kwargs.get("research_goal", "")
        system_description = kwargs.get("system_description", "")
        
        # Extract simulation parameters
        ensemble = kwargs.get("ensemble", "NPT")
        simulation_time = kwargs.get("simulation_time", 2.0)
        temperature = kwargs.get("temperature", 300.0)
        pressure = kwargs.get("pressure", 1.0)
        timestep = kwargs.get("timestep", 2.0)
        required_outputs = kwargs.get("required_outputs", ["energy", "trajectory"])
        properties_to_calculate = kwargs.get("properties_to_calculate", [])
        
        # Calculate runtime steps
        runtime_steps = int((simulation_time * 1e6) / timestep)
        equil_steps = int((kwargs.get("equilibration_time", simulation_time * 0.25) * 1e6) / timestep)
        prod_steps = int((kwargs.get("production_time", simulation_time * 0.75) * 1e6) / timestep)
        
        # Create element information string
        element_info = []
        for element, count in system_info.get("element_counts", {}).items():
            element_info.append(f"{element}: {count} atoms")
        element_info_str = "\n  - ".join(element_info) if element_info else "Unknown"
        
        # Include bond and angle information
        bond_types = system_info.get("bond_types", 0)
        angle_types = system_info.get("angle_types", 0)
        
        # Generate outputs section based on required outputs
        output_commands = self._generate_output_commands(required_outputs, properties_to_calculate, system_info)
        
       # Format the template with actual values
        prompt = LAMMPS_INPUT_GENERATION_TEMPLATE.format(
            research_goal=research_goal,
            system_description=system_description,
            element_info_str=element_info_str,
            atom_count=system_info.get('atom_count', 0),
            box_dimensions=system_info.get('box_dimensions', [40, 40, 40]),
            bond_types=bond_types,
            angle_types=angle_types,
            has_water="Yes" if system_info.get('has_water', False) else "No",
            has_ions="Yes" if system_info.get('has_ions', False) else "No",
            has_organic="Yes" if system_info.get('has_organic', False) else "No",
            properties_to_calculate_str=", ".join(properties_to_calculate),
            required_outputs_str=", ".join(required_outputs),
            temperature=temperature,
            pressure=pressure,
            ensemble=ensemble,
            timestep=timestep,
            simulation_time=simulation_time,
            equil_steps=equil_steps,
            prod_steps=prod_steps,
            data_filename=data_filename,
            output_commands=output_commands) 
        response = self.model.generate_content(prompt)
        script_text = response.text
        
        # Clean the script output
        script_text = self._clean_script(script_text)
        
        return script_text
    
    def _generate_output_commands(self, 
                                required_outputs: List[str], 
                                properties_to_calculate: List[str], 
                                system_info: Dict[str, Any]) -> str:
        """
        Generate LAMMPS output command instructions based on required outputs and properties.
        """
        instructions = []
        
        # Basic outputs that should always be included
        instructions.append("Include regular thermodynamic output (temperature, pressure, energy, etc.)")
        
        # Trajectory output
        if "trajectory" in required_outputs:
            instructions.append("Output trajectory in DCD or XYZ format at appropriate intervals")
        
        # Density
        if "density" in required_outputs or "density" in properties_to_calculate:
            instructions.append("Calculate and output system density")
        
        # RDF
        if "rdf" in required_outputs or any(p in properties_to_calculate for p in ["rdf", "radial distribution", "pair correlation"]):
            atom_pairs = []
            if system_info.get("has_water", False):
                atom_pairs.append("O-O")
            if system_info.get("has_ions", False):
                if "Na" in system_info.get("elements", []) and "Cl" in system_info.get("elements", []):
                    atom_pairs.append("Na-Cl")
                    atom_pairs.append("Na-O")
                    atom_pairs.append("Cl-O")
            
            if atom_pairs:
                instructions.append(f"Calculate radial distribution functions for atom pairs: {', '.join(atom_pairs)}")
            else:
                instructions.append("Calculate radial distribution functions for relevant atom pairs")
        
        # MSD/Diffusion
        if "msd" in required_outputs or "diffusion" in required_outputs or any(p in properties_to_calculate for p in ["diffusion", "mobility", "msd"]):
            if system_info.get("has_ions", False):
                instructions.append("Calculate mean squared displacement (MSD) separately for each ion type")
            else:
                instructions.append("Calculate mean squared displacement (MSD) for appropriate atom types")
        
        # Viscosity
        if "viscosity" in required_outputs or "viscosity" in properties_to_calculate:
            instructions.append("Calculate viscosity using Green-Kubo formalism with pressure tensor autocorrelation")
        
        # Dielectric
        if "dielectric" in required_outputs or any(p in properties_to_calculate for p in ["dielectric", "polarization"]):
            instructions.append("Track system dipole moment for dielectric constant calculation")
        
        return "\n".join(instructions)
    
    def _clean_script(self, script_text: str) -> str:
        """Remove markdown formatting and other unwanted elements from the script."""
        # Remove markdown code block markers
        script_text = re.sub(r'```(?:lammps|bash)?', '', script_text)
        
        # Remove any trailing backticks that might be closing a code block
        script_text = script_text.replace('```', '')
        
        # Remove any leading/trailing whitespace
        script_text = script_text.strip()
        
        # Ensure the script starts with a valid LAMMPS command or comment
        if not script_text.startswith(('#', 'units', 'echo', 'log', 'atom_style')):
            # Add a comment at the beginning if needed
            script_text = f"# LAMMPS script for: {script_text.split(os.linesep)[0]}\n\n" + script_text
        
        self.logger.info("Cleaned script output of markdown formatting")
        return script_text
    
    def _ensure_force_field_parameters(self, script: str, system_info: Dict[str, Any]) -> str:
        """Ensure the script has all necessary force field parameters."""
        lines = script.split('\n')
        
        # Check if script has necessary force field commands
        has_bond_style = any("bond_style" in line.lower() for line in lines)
        has_bond_coeffs = any("bond_coeff" in line.lower() for line in lines)
        has_angle_style = any("angle_style" in line.lower() for line in lines)
        has_angle_coeffs = any("angle_coeff" in line.lower() for line in lines)
        
        # If missing bond or angle parameters, add them
        if not (has_bond_style and has_bond_coeffs and has_angle_style and has_angle_coeffs):
            self.logger.warning("Adding missing force field parameters to the script")
            
            # Find insert position - after read_data but before fixes
            insert_idx = 0
            for i, line in enumerate(lines):
                if "read_data" in line:
                    insert_idx = i + 1
                    break
            
            # Generate force field parameters
            ff_params = self._generate_force_field_parameters(system_info)
            
            # Insert force field parameters
            lines.insert(insert_idx, "\n# Force field parameters added by LAMMPSSimulationAgent")
            lines.insert(insert_idx + 1, ff_params)
            lines.insert(insert_idx + 2, "")
            
        # Return the updated script
        return '\n'.join(lines)
    
    def _generate_force_field_parameters(self, system_info: Dict[str, Any]) -> str:
        """Generate force field parameters based on system analysis."""
        params = []
        
        # Add basic styles
        params.append("# Basic force field styles")
        params.append("pair_style lj/cut/coul/long 10.0")
        params.append("bond_style harmonic")
        params.append("angle_style harmonic")
        params.append("special_bonds lj/coul 0.0 0.0 0.5")
        params.append("kspace_style pppm 0.0001")
        params.append("")
        
        # Add bond coefficients
        bond_types = system_info.get("bond_types", 0)
        if bond_types > 0:
            params.append("# Bond coefficients")
            for i in range(1, bond_types + 1):
                params.append(f"bond_coeff {i} 450.0 1.0  # Generic bond")
            params.append("")
        
        # Add angle coefficients
        angle_types = system_info.get("angle_types", 0)
        if angle_types > 0:
            params.append("# Angle coefficients")
            for i in range(1, angle_types + 1):
                params.append(f"angle_coeff {i} 55.0 109.47  # Generic angle")
            params.append("")
        
        # Add pair coefficients for common elements
        params.append("# Pair coefficients")
        element_types = {}
        type_idx = 1
        
        for element in system_info.get("elements", []):
            element_types[element] = type_idx
            type_idx += 1
            
        # If elements were found, add pair coefficients
        if element_types:
            for el, idx in element_types.items():
                if el == "O":
                    params.append(f"pair_coeff {idx} {idx} 0.1553 3.166  # Oxygen")
                elif el == "H":
                    params.append(f"pair_coeff {idx} {idx} 0.0 0.0  # Hydrogen")
                elif el == "Na":
                    params.append(f"pair_coeff {idx} {idx} 0.0115 2.275  # Sodium")
                elif el == "Cl":
                    params.append(f"pair_coeff {idx} {idx} 0.1 4.417  # Chloride")
                else:
                    params.append(f"pair_coeff {idx} {idx} 0.1 3.0  # Generic {el}")
        # If no elements were found, add generic coefficients
        else:
            params.append("pair_coeff * * 0.0 0.0")
            params.append("# For water O-O")
            params.append("pair_coeff 1 1 0.1553 3.166")
            
        return "\n".join(params)
    
    def _generate_readme(self, **kwargs) -> str:
        """Generate a README file with analysis instructions based on the research goal."""
        research_goal = kwargs.get("research_goal", "")
        system_description = kwargs.get("system_description", "")
        system_info = kwargs.get("system_info", {})
        simulation_params = kwargs.get("simulation_params", {})
        
        # Create README path
        readme_path = os.path.join(self.working_dir, "README.md")
        
        # Generate content based on research goal and simulation parameters
        with open(readme_path, 'w') as f:
            # Header
            f.write(f"# Molecular Dynamics Simulation: {system_description}\n\n")
            
            # Research goal
            f.write(f"## Research Goal\n{research_goal}\n\n")
            
            # System composition
            f.write("## System Composition\n")
            for element, count in system_info.get("element_counts", {}).items():
                f.write(f"- {element}: {count} atoms\n")
            f.write(f"- Total atoms: {system_info.get('atom_count', 'Unknown')}\n\n")
            
            # Simulation parameters
            f.write("## Simulation Parameters\n")
            properties = simulation_params.get("properties_to_calculate", [])
            if properties:
                f.write(f"- Properties to calculate: {', '.join(properties)}\n")
            f.write(f"- Temperature: {simulation_params.get('temperature', 300.0)} K\n")
            f.write(f"- Pressure: {simulation_params.get('pressure', 1.0)} atm\n")
            f.write(f"- Ensemble: {simulation_params.get('ensemble', 'NPT')}\n")
            f.write(f"- Timestep: {simulation_params.get('timestep', 2.0)} fs\n")
            f.write(f"- Total simulation time: {simulation_params.get('simulation_time', 2.0)} ns\n\n")
            
            # How to run
            f.write("## How to Run\n")
            f.write("```bash\n")
            f.write(f"cd {self.working_dir}\n")
            f.write(f"lmp -in {os.path.basename(kwargs.get('script_path', 'run.lammps'))}\n")
            f.write("```\n\n")
            
            # Analysis instructions
            f.write("## Analysis Instructions\n")
            
            # Generate specific analysis instructions based on properties to calculate
            analysis_steps = self._generate_analysis_instructions(
                research_goal=research_goal,
                properties=properties,
                required_outputs=simulation_params.get("required_outputs", []),
                system_info=system_info
            )
            
            for i, step in enumerate(analysis_steps, 1):
                f.write(f"{i}. {step}\n")
            
        return readme_path
    
    def _generate_analysis_instructions(self, 
                                      research_goal: str, 
                                      properties: List[str], 
                                      required_outputs: List[str],
                                      system_info: Dict[str, Any]) -> List[str]:
        """Generate step-by-step analysis instructions based on the research goal."""
        # Start with standard instructions
        instructions = [
            "Verify system equilibration by checking energy, temperature, and pressure over time",
            "Analyze trajectory files using visualization tools like VMD or OVITO"
        ]
        
        # Add property-specific instructions
        if "density" in properties or "density" in required_outputs:
            instructions.append("Calculate average density from the production phase")
            instructions.append("Compare density with experimental values")
        
        if any(p in properties + required_outputs for p in ["diffusion", "msd", "mobility"]):
            instructions.append("Plot mean squared displacement (MSD) vs time")
            instructions.append("Calculate diffusion coefficients using the Einstein relation: D = MSD/(6t)")
            if system_info.get("has_ions", False):
                instructions.append("Compare diffusion coefficients of different ion types")
        
        if any(p in properties + required_outputs for p in ["rdf", "structure", "radial"]):
            instructions.append("Plot radial distribution functions to analyze molecular structure")
            instructions.append("Identify coordination shells from RDF peaks")
            if system_info.get("has_water", False) and system_info.get("has_ions", False):
                instructions.append("Calculate hydration numbers of ions by integrating the first peak of ion-water RDFs")
        
        if any(p in properties + required_outputs for p in ["viscosity"]):
            instructions.append("Calculate viscosity from the Green-Kubo integral of pressure tensor autocorrelation")
            instructions.append("Compare calculated viscosity with experimental values")
        
        if any(p in properties + required_outputs for p in ["dielectric", "polarization"]):
            instructions.append("Calculate dielectric constant from dipole moment fluctuations")
        
        # Add customized instructions based on research goal
        lower_goal = research_goal.lower()
        
        if "compare" in lower_goal or "different" in lower_goal:
            instructions.append("Compare results across different simulation conditions or systems")
        
        if "temperature" in lower_goal and "effect" in lower_goal:
            instructions.append("Plot the calculated properties as a function of temperature to identify trends")
        
        if "pressure" in lower_goal and "effect" in lower_goal:
            instructions.append("Plot the calculated properties as a function of pressure to identify trends")
        
        if "concentration" in lower_goal:
            instructions.append("Analyze how properties change with concentration")
        
        return instructions
