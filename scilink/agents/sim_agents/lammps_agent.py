import os
import re
import shutil
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from ase import io
from ase.io.lammpsdata import read_lammps_data

class LAMMPSSimulationAgent:
    def __init__(self, working_dir: str, api_key: Optional[str] = None):
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
    
    def analyze_system(self, data_file: str) -> Dict[str, Any]:
        """
        Analyze a LAMMPS data file using ASE to identify its components.
        """
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
                        element_counts['H'] >= 2 * element_counts['O'])
            has_ions = any(x in element_counts for x in ['Na', 'Cl', 'K', 'Ca', 'Mg'])
            
            # Get bond and angle information from the data file
            bond_types, angle_types = self._extract_bond_angle_types(data_file)
            
            system_info = {
                "atom_count": len(atoms),
                "elements": list(element_counts.keys()),
                "element_counts": element_counts,
                "box_dimensions": atoms.get_cell().diagonal(),
                "has_water": has_water,
                "has_ions": has_ions,
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
                "bond_types": 0,
                "angle_types": 0
            }
    
    def _extract_bond_angle_types(self, data_file: str) -> tuple:
        """
        Extract bond and angle type information from a LAMMPS data file.
        
        Returns:
            tuple: (bond_types, angle_types)
        """
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
    
    def generate_simulation_protocol(self, 
                                   data_file: str, 
                                   system_description: str,
                                   temperature: float = 300.0, 
                                   pressure: float = 1.0,
                                   simulation_time: float = 1.0,  # in nanoseconds
                                   ensemble: str = "NPT",
                                   **kwargs):
        """
        Generate a LAMMPS simulation protocol for the given system.
        """
        # Copy the data file to the working directory
        local_data_file = self._copy_data_file(data_file)
        
        # Analyze system with ASE
        system_info = self.analyze_system(data_file)
        
        # Combine with description-based analysis for robustness
        if not system_info["elements"]:
            self.logger.warning("Using description-based system analysis as fallback")
            system_info["has_water"] = system_info["has_water"] or "water" in system_description.lower()
            system_info["has_ions"] = system_info["has_ions"] or any(x in system_description.lower() for x in ["nacl", "salt", "ion"])
        
        # Identify additional properties from description
        system_info["has_proteins"] = "protein" in system_description.lower()
        system_info["has_lipids"] = "lipid" in system_description.lower() or "membrane" in system_description.lower()
        system_info["has_polymers"] = "polymer" in system_description.lower()
        
        # Extract timestep from kwargs or use default
        timestep = kwargs.pop("timestep", 2.0)  # Remove from kwargs
        
        # Calculate runtime in timesteps
        runtime_steps = int((simulation_time * 1e6) / timestep)  # Convert ns to timesteps
        
        # Use LLM to generate custom LAMMPS script
        simulation_script = self._generate_lammps_script_with_llm(
            data_filename=os.path.basename(local_data_file),
            system_description=system_description,
            system_info=system_info,
            temperature=temperature,
            pressure=pressure,
            simulation_time=simulation_time,
            ensemble=ensemble,
            runtime_steps=runtime_steps,
            timestep=timestep,
            **kwargs
        )
        
        # Validate and add force field parameters if needed
        simulation_script = self._ensure_force_field_parameters(simulation_script, system_info)
        
        # Save the generated simulation script
        script_path = os.path.join(self.working_dir, "run.lammps")
        with open(script_path, 'w') as f:
            f.write(simulation_script)
        
        self.logger.info(f"Generated LAMMPS script saved to {script_path}")
        return {
            "script_path": script_path, 
            "system_info": system_info,
            "data_file": local_data_file
        }
    
    def _copy_data_file(self, source_path: str) -> str:
        """
        Copy the data file to the working directory.
        
        Args:
            source_path: Path to the source data file
            
        Returns:
            Path to the copied data file in the working directory
        """
        # Get the filename from the source path
        filename = os.path.basename(source_path)
        
        # Default to "system.data" if we want a consistent name
        dest_filename = "system.data"
        dest_path = os.path.join(self.working_dir, dest_filename)
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        self.logger.info(f"Copied data file from {source_path} to {dest_path}")
        
        return dest_path
    
    def _generate_lammps_script_with_llm(self, **kwargs):
        """Generate a LAMMPS script using LLM and clean the output."""
        system_info = kwargs.get("system_info", {})
        data_filename = kwargs.get("data_filename", "system.data")
        
        # Create element information string
        element_info = []
        for element, count in system_info.get("element_counts", {}).items():
            element_info.append(f"{element}: {count} atoms")
        element_info_str = "\n  - ".join(element_info) if element_info else "Unknown"
        
        # Include bond and angle information
        bond_types = system_info.get("bond_types", 0)
        angle_types = system_info.get("angle_types", 0)
        
        prompt = f"""
        Generate a complete LAMMPS simulation script for the following system:
        
        SYSTEM DESCRIPTION: {kwargs.get('system_description', '')}
        
        SYSTEM COMPOSITION:
          - {element_info_str}
          - Total atoms: {system_info.get('atom_count', 'Unknown')}
          - Box dimensions: {system_info.get('box_dimensions', [40, 40, 40])}
          - Bond types: {bond_types}
          - Angle types: {angle_types}
          
        DETECTED COMPONENTS:
          - Water: {'Yes' if system_info.get('has_water', False) else 'No'}
          - Ions: {'Yes' if system_info.get('has_ions', False) else 'No'}
          - Proteins: {'Yes' if system_info.get('has_proteins', False) else 'No'}
          - Lipids: {'Yes' if system_info.get('has_lipids', False) else 'No'}
          - Polymers: {'Yes' if system_info.get('has_polymers', False) else 'No'}
        
        SIMULATION PARAMETERS:
          - Temperature: {kwargs.get('temperature', 300.0)} K
          - Pressure: {kwargs.get('pressure', 1.0)} atm
          - Simulation time: {kwargs.get('simulation_time', 1.0)} ns
          - Ensemble: {kwargs.get('ensemble', 'NPT')}
          - Timestep: {kwargs.get('timestep', 2.0)} fs
          
        IMPORTANT REQUIREMENTS:
        1. Proper initialization (units real, etc.)
        2. Reading the data file "{data_filename}"
        3. COMPLETE force field settings including:
           - Bond style and bond coefficients for ALL bond types
           - Angle style and angle coefficients for ALL angle types
           - Pair style and pair coefficients for ALL atom types
        4. Energy minimization
        5. Equilibration phase(s)
        6. Production run for {kwargs.get('runtime_steps', 500000)} timesteps
        7. Output trajectory and analysis data
        8. If water is present, include proper SHAKE constraints with appropriate bond and angle types
        
        EXAMPLE FORCE FIELD PARAMETERS (for water and ions):
        ```
        # Force field definitions
        pair_style lj/cut/coul/long 10.0
        bond_style harmonic
        angle_style harmonic
        
        # Bond coefficients (for SPCE water)
        bond_coeff 1 1000.0 1.0  # O-H bond
        
        # Angle coefficients (for SPCE water)
        angle_coeff 1 100.0 109.47  # H-O-H angle
        
        # Pair coefficients
        pair_coeff 1 1 0.1553 3.166  # O-O
        pair_coeff 2 2 0.0 0.0       # H-H
        pair_coeff 3 3 0.0115 2.275  # Na-Na
        pair_coeff 4 4 0.1 4.417     # Cl-Cl
        ```
        
        IMPORTANT: Do NOT include markdown formatting, code block markers, or any backticks in your response.
        Return ONLY the raw LAMMPS script content without any additional text.
        """
        
        response = self.model.generate_content(prompt)
        script_text = response.text
        
        # Clean the script output
        script_text = self._clean_script(script_text)
        
        return script_text
    
    def _ensure_force_field_parameters(self, script: str, system_info: Dict[str, Any]) -> str:
        """
        Ensure the script has all necessary force field parameters.
        
        Args:
            script: The LAMMPS script
            system_info: System information dictionary
            
        Returns:
            Updated LAMMPS script with force field parameters
        """
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
        """
        Generate force field parameters based on system analysis.
        
        Args:
            system_info: System information dictionary
            
        Returns:
            String containing force field parameter commands
        """
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
    
    def _clean_script(self, script_text: str) -> str:
        """Remove markdown formatting and other unwanted elements from the script."""
        import re
        
        # Remove markdown code block markers
        script_text = re.sub(r'```(?:lammps|bash)?', '', script_text)
        
        # Remove any trailing backticks that might be closing a code block
        script_text = script_text.replace('```', '')
        
        # Remove any leading/trailing whitespace
        script_text = script_text.strip()
        
        # Ensure the script starts with a valid LAMMPS command or comment
        if not script_text.startswith(('#', 'units', 'echo', 'log', 'atom_style')):
            # Add a comment at the beginning if needed
            script_text = "# LAMMPS simulation script generated by LAMMPSSimulationAgent\n\n" + script_text
        
        self.logger.info("Cleaned script output of markdown formatting")
        return script_text
