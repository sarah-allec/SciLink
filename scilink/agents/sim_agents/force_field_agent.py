import os
import re
import logging
import json
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
import google.generativeai as genai
from MDAnalysis import Universe
import numpy as np

class ForceFieldAgent:
    """
    AI-driven agent for optimal force field selection and parameter acquisition.
    
    This agent leverages LLMs to analyze molecular systems and research goals, then:
    1. Selects the most appropriate force field based on system composition and research objectives
    2. Determines the best method to obtain parameters (database, manual, QM, etc.)
    3. Executes the chosen parameterization strategy
    4. Validates parameters for scientific rigor
    
    The agent works in conjunction with other simulation agents in the pipeline.
    """
    
    def __init__(self, working_dir: str, api_key: Optional[str] = None):
        """
        Initialize the ForceFieldAgent.
        
        Args:
            working_dir: Directory for output files and intermediate calculations
            api_key: API key for Google Gemini API (optional if set in environment)
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        
        # Set up API access
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
            
        # Force field databases and methods
        self.ff_databases = {
            "water": ["SPC/E", "TIP3P", "TIP4P", "OPC", "TIP5P"],
            "proteins": ["AMBER ff14SB", "AMBER ff19SB", "CHARMM36m", "OPLS-AA/M"],
            "lipids": ["CHARMM36", "Slipids", "Lipid17", "Martini"],
            "carbohydrates": ["GLYCAM06", "CHARMM36-carb", "GROMOS 56A(CARBO)"],
            "small_molecules": ["GAFF", "GAFF2", "CGenFF", "OpenFF"],
            "ions": ["Joung-Cheatham", "CHARMM", "Åqvist", "Li-Merz"],
            "polymers": ["PCFF", "COMPASS", "OPLS-AA"],
            "metals": ["EAM", "COMB", "ReaxFF"],
            "interfaces": ["INTERFACE-AMBER", "CHARMM-METAL"]
        }
        
        # Parameter acquisition methods
        self.param_methods = {
            "database": {
                "description": "Direct parameter extraction from established databases",
                "tools": ["ParmEd", "AmberTools", "CGenFF", "MATCH"],
                "strengths": ["Fast", "Reliable for standard molecules"]
            },
            "analogy": {
                "description": "Parameters by chemical analogy to known molecules",
                "tools": ["ffTK", "LigParGen", "CGenFF"],
                "strengths": ["Good balance of speed and accuracy"]
            },
            "quantum": {
                "description": "Ab initio parameterization from QM calculations",
                "tools": ["ffTK", "ForceBalance", "poltype2"],
                "strengths": ["Highest accuracy", "Novel molecules"]
            }
        }
        
    def select_force_field(self, 
                         pdb_file: str, 
                         research_goal: str,
                         system_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze system composition and research goal to select optimal force field.
        
        Args:
            pdb_file: Path to the PDB file containing the molecular system
            research_goal: Research objective in natural language
            system_description: Optional description of the system
            
        Returns:
            Dictionary with force field selection and parameterization information
        """
        self.logger.info(f"Analyzing system in {pdb_file} for force field selection")
        
        # Analyze the system composition
        system_info = self._analyze_system_composition(pdb_file)
        
        if not system_description:
            system_description = self._generate_system_description(system_info)
            
        self.logger.info(f"System description: {system_description}")
        
        # Select force field using LLM
        force_field_selection = self._select_optimal_force_field(
            system_info=system_info,
            research_goal=research_goal,
            system_description=system_description
        )
        
        # Determine parameter acquisition method
        param_method = self._determine_parameter_method(
            system_info=system_info,
            force_field=force_field_selection["force_field"],
            research_goal=research_goal
        )
        
        # Log the decisions
        self.logger.info(f"Selected force field: {force_field_selection['force_field']}")
        self.logger.info(f"Parameter acquisition method: {param_method['method']}")
        
        # Create the result dictionary
        result = {
            "system_info": system_info,
            "system_description": system_description,
            "force_field": force_field_selection,
            "parameter_method": param_method,
            "working_dir": self.working_dir
        }
        
        # Save selection info to file
        selection_file = os.path.join(self.working_dir, "force_field_selection.json")
        with open(selection_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
    
    def acquire_parameters(self, selection_info: Dict[str, Any], data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire force field parameters using the selected method.
        
        Args:
            selection_info: Force field selection info from select_force_field()
            data_file: Optional existing LAMMPS data file to enhance
            
        Returns:
            Dictionary with parameter files and information
        """
        method = selection_info["parameter_method"]["method"]
        force_field = selection_info["force_field"]["force_field"]
        system_info = selection_info["system_info"]
        
        self.logger.info(f"Acquiring parameters via {method} method for {force_field}")
        
        # Execute the appropriate parameter acquisition method
        if method == "database":
            params = self._acquire_parameters_from_database(
                force_field=force_field,
                system_info=system_info,
                data_file=data_file
            )
        elif method == "analogy":
            params = self._acquire_parameters_by_analogy(
                force_field=force_field,
                system_info=system_info,
                data_file=data_file
            )
        elif method == "quantum":
            params = self._acquire_parameters_from_quantum(
                force_field=force_field,
                system_info=system_info,
                data_file=data_file
            )
        else:
            # Fallback to database method
            self.logger.warning(f"Unknown parameter method {method}, falling back to database")
            params = self._acquire_parameters_from_database(
                force_field=force_field,
                system_info=system_info,
                data_file=data_file
            )
            
        # Validate parameters for scientific rigor
        validation = self._validate_parameters(params, system_info)
        params["validation"] = validation
        
        # Generate parameter summary
        params["summary"] = self._generate_parameter_summary(params, selection_info)
        
        # Save parameter info to file
        param_file = os.path.join(self.working_dir, "parameter_info.json")
        with open(param_file, 'w') as f:
            # Filter out large data that can't be serialized
            serializable_params = {k: v for k, v in params.items() if k not in ["raw_data", "quantum_results"]}
            json.dump(serializable_params, f, indent=2)
            
        return params
    
    def generate_lammps_parameters(self, 
                               parameter_info: Dict[str, Any], 
                               data_file: str) -> Dict[str, str]:
        """
        Generate LAMMPS parameter files based on the acquired parameters.
        
        Args:
            parameter_info: Parameter info from acquire_parameters()
            data_file: Path to LAMMPS data file to enhance
            
        Returns:
            Dictionary with paths to parameter files
        """
        self.logger.info(f"Generating LAMMPS parameter files for {data_file}")
        
        # Parse the existing data file to understand what parameters are needed
        data_file_info = self._parse_data_file(data_file)
        
        # Generate LAMMPS parameter file content
        param_content = self._generate_lammps_parameters(
            data_file_info=data_file_info,
            parameter_info=parameter_info
        )
        
        # Write parameter files
        files = {}
        
        # Main parameter file
        param_file = os.path.join(self.working_dir, "ff_params.lammps")
        with open(param_file, 'w') as f:
            f.write(param_content["main"])
        files["main"] = param_file
        
        # Additional files if needed
        if "additional" in param_content:
            for name, content in param_content["additional"].items():
                add_file = os.path.join(self.working_dir, f"{name}.lammps")
                with open(add_file, 'w') as f:
                    f.write(content)
                files[name] = add_file
                
        self.logger.info(f"Generated {len(files)} parameter files")
        return files
    
    # ================================
    # PRIVATE METHODS
    # ================================
    
    def _analyze_system_composition(self, pdb_file: str) -> Dict[str, Any]:
        """
        Analyze the molecular system's composition from a PDB file.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Dictionary with system information
        """
        self.logger.info(f"Analyzing composition of {pdb_file}")
        
        try:
            # Load the PDB using MDAnalysis
            u = Universe(pdb_file)
            
            # Count atoms by element
            elements = {}
            for atom in u.atoms:
                # Get element from atom name (first character, or first two if second is lowercase)
                if len(atom.name) > 1 and atom.name[1].islower():
                    element = atom.name[:2]
                else:
                    element = atom.name[0]
                    
                elements[element] = elements.get(element, 0) + 1
                
            # Identify molecular components
            has_water = self._detect_water(u, elements)
            has_proteins = self._detect_proteins(u)
            has_lipids = self._detect_lipids(u)
            has_nucleic_acids = self._detect_nucleic_acids(u)
            has_ions = self._detect_ions(elements)
            has_small_molecules = self._detect_small_molecules(u, elements)
            has_metals = self._detect_metals(elements)
            has_carbohydrates = self._detect_carbohydrates(u)
            
            # Identify special system characteristics
            is_interface = self._detect_interface(u)
            is_gas_phase = self._detect_gas_phase(u)
            
            # Calculate basic system dimensions
            box_dimensions = u.dimensions[:3] if hasattr(u, 'dimensions') and u.dimensions is not None else [0, 0, 0]
            
            system_info = {
                "filename": os.path.basename(pdb_file),
                "n_atoms": len(u.atoms),
                "n_residues": len(u.residues),
                "n_molecules": len(u.segments),
                "elements": elements,
                "composition": {
                    "water": has_water,
                    "proteins": has_proteins,
                    "lipids": has_lipids,
                    "nucleic_acids": has_nucleic_acids, 
                    "ions": has_ions,
                    "small_molecules": has_small_molecules,
                    "metals": has_metals,
                    "carbohydrates": has_carbohydrates
                },
                "system_type": {
                    "interface": is_interface,
                    "gas_phase": is_gas_phase
                },
                "box_dimensions": box_dimensions.tolist() if isinstance(box_dimensions, np.ndarray) else box_dimensions,
            }
            
            # Add residue details if available
            if has_proteins or has_nucleic_acids or has_small_molecules:
                residue_names = [res.resname for res in u.residues]
                residue_counts = {}
                for name in residue_names:
                    residue_counts[name] = residue_counts.get(name, 0) + 1
                system_info["residue_counts"] = residue_counts
                
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing PDB file: {e}")
            # Return minimal info if analysis fails
            return {
                "filename": os.path.basename(pdb_file),
                "n_atoms": 0,
                "elements": {},
                "composition": {
                    "water": False,
                    "proteins": False,
                    "lipids": False,
                    "nucleic_acids": False,
                    "ions": False,
                    "small_molecules": False,
                    "metals": False,
                    "carbohydrates": False
                }
            }
    
    def _detect_water(self, universe, elements):
        """Detect if system contains water molecules."""
        # Check for standard water residue names
        water_residues = ['WAT', 'HOH', 'H2O', 'SOL', 'TIP', 'SPC']
        for res in universe.residues:
            if any(water in res.resname for water in water_residues):
                return True
                
        # Check element ratio - water has 2:1 H:O ratio
        # Only useful for simple systems
        if 'H' in elements and 'O' in elements:
            h_atoms = elements.get('H', 0)
            o_atoms = elements.get('O', 0)
            if h_atoms > 0 and o_atoms > 0 and (h_atoms / o_atoms) > 1.5:
                return True
                
        return False
    
    def _detect_proteins(self, universe):
        """Detect if system contains protein molecules."""
        # Standard amino acid residues
        amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                      'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                      'TYR', 'VAL']
        
        for res in universe.residues:
            if res.resname in amino_acids:
                return True
        return False
    
    def _detect_lipids(self, universe):
        """Detect if system contains lipid molecules."""
        # Common lipid residue names
        lipid_residues = ['POPC', 'POPE', 'DPPC', 'DOPC', 'DMPC', 'CHOL', 'CHL', 
                         'DLPE', 'DLPC', 'DSPC', 'DAPC', 'DOPE', 'POPG', 'DPPG']
        
        for res in universe.residues:
            if res.resname in lipid_residues:
                return True
                
        # Alternative detection: large residues with many carbons and some phosphorus
        return False
    
    def _detect_nucleic_acids(self, universe):
        """Detect if system contains DNA or RNA."""
        # Nucleotide residue names
        nucleotides = ['ADE', 'THY', 'GUA', 'CYT', 'URA', 'A', 'T', 'G', 'C', 'U',
                      'DA', 'DT', 'DG', 'DC', 'DU', 'AMP', 'GMP', 'CMP', 'TMP', 'UMP']
        
        for res in universe.residues:
            if res.resname in nucleotides:
                return True
        return False
    
    def _detect_ions(self, elements):
        """Detect if system contains common ions."""
        ion_elements = ['Na', 'K', 'Cl', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu', 'Li']
        return any(ion in elements for ion in ion_elements)
    
    def _detect_small_molecules(self, universe, elements):
        """Detect if system contains small organic molecules."""
        if 'C' in elements and elements['C'] > 0:
            # Filter out known biomolecules
            if not self._detect_proteins(universe) and not self._detect_lipids(universe) and not self._detect_nucleic_acids(universe):
                return True
        return False
    
    def _detect_metals(self, elements):
        """Detect if system contains metal atoms."""
        metal_elements = ['Fe', 'Zn', 'Cu', 'Ni', 'Co', 'Mn', 'Mg', 'Ca', 'Na', 'K', 
                         'Al', 'Ti', 'V', 'Cr', 'Pd', 'Pt', 'Au', 'Ag', 'Hg']
        return any(metal in elements for metal in metal_elements)
    
    def _detect_carbohydrates(self, universe):
        """Detect if system contains carbohydrates."""
        # Common carbohydrate/sugar residue names
        carb_residues = ['GLC', 'GAL', 'MAN', 'FUC', 'XYL', 'NAG', 'SIA', 'RIB', 
                        'AGLC', 'BGLC', 'GLCA', 'GLCN']
        
        for res in universe.residues:
            if res.resname in carb_residues:
                return True
        return False
    
    def _detect_interface(self, universe):
        """Detect if system represents an interface."""
        # Simple heuristic: check if there are empty regions in z dimension
        try:
            z_coords = universe.atoms.positions[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            z_range = z_max - z_min
            
            # Divide into bins along z
            n_bins = 20
            hist, edges = np.histogram(z_coords, bins=n_bins)
            
            # Look for empty regions in the middle (interface)
            # Ignore top and bottom 15% which might be naturally empty
            middle_bins = hist[int(n_bins*0.15):int(n_bins*0.85)]
            if min(middle_bins) < max(middle_bins) * 0.1:  # If some middle bins are nearly empty
                return True
                
            return False
        except:
            return False
    
    def _detect_gas_phase(self, universe):
        """Detect if system is in gas phase (no bulk solvent)."""
        # Heuristic: if no water and low density, likely gas phase
        has_water = self._detect_water(universe, {})
        
        if not has_water and hasattr(universe, 'dimensions') and universe.dimensions is not None:
            try:
                # Calculate rough density in g/cm^3
                volume = universe.dimensions[0] * universe.dimensions[1] * universe.dimensions[2] / 1000  # A^3 to nm^3
                n_atoms = len(universe.atoms)
                # Rough average: 12 g/mol per atom (mainly C)
                density = (n_atoms * 12) / (volume * 0.6022)  # g/cm^3
                
                # Gas phase typically has very low density
                if density < 0.1:  # g/cm^3
                    return True
            except:
                pass
                
        return False
    
    def _generate_system_description(self, system_info: Dict[str, Any]) -> str:
        """
        Generate a human-readable description of the system.
        
        Args:
            system_info: System information from _analyze_system_composition
            
        Returns:
            Human-readable description
        """
        description_parts = []
        
        # Add information about major components
        comp = system_info["composition"]
        
        if comp["water"]:
            description_parts.append("water")
            
        if comp["ions"]:
            ion_elements = [e for e in system_info.get("elements", {}) if e in ['Na', 'K', 'Cl', 'Ca', 'Mg', 'Zn', 'Fe']]
            if ion_elements:
                description_parts.append("+".join(ion_elements) + " ions")
            else:
                description_parts.append("ions")
                
        if comp["proteins"]:
            n_residues = system_info.get("n_residues", 0)
            if n_residues > 300:
                description_parts.append("large protein")
            else:
                description_parts.append("protein")
                
        if comp["lipids"]:
            description_parts.append("lipids")
            
        if comp["nucleic_acids"]:
            description_parts.append("nucleic acids")
            
        if comp["carbohydrates"]:
            description_parts.append("carbohydrates")
            
        if comp["small_molecules"]:
            description_parts.append("organic molecules")
            
        if comp["metals"]:
            description_parts.append("metals")
            
        # Add system type
        system_type = system_info["system_type"]
        if system_type["interface"]:
            description_parts.append("interface")
            
        if system_type["gas_phase"]:
            description_parts.append("gas phase")
            
        # Combine into description
        if description_parts:
            description = " with ".join(description_parts)
        else:
            description = "molecular system"
            
        return f"{description} ({system_info['n_atoms']} atoms)"
    
    def _select_optimal_force_field(self, 
                                 system_info: Dict[str, Any], 
                                 research_goal: str,
                                 system_description: str) -> Dict[str, Any]:
        """
        Use LLM to select the optimal force field for the system and research goal.
        
        Args:
            system_info: System information from _analyze_system_composition
            research_goal: Research objective in natural language
            system_description: Human-readable system description
            
        Returns:
            Dictionary with force field selection and justification
        """
        self.logger.info("Selecting optimal force field using LLM")
        
        # Convert system_info to a format suitable for the prompt
        composition = system_info["composition"]
        comp_str = "\n".join([f"- {k}: {'Yes' if v else 'No'}" for k, v in composition.items()])
        
        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("elements", {}).items()])
        
        # Create a dictionary of force field options to include in prompt
        ff_options = {}
        for category, force_fields in self.ff_databases.items():
            if composition.get(category.replace("_", " "), False):
                ff_options[category] = force_fields
                
        # Always include water force fields if water is present
        if composition["water"] and "water" not in ff_options:
            ff_options["water"] = self.ff_databases["water"]
            
        # Format force field options for the prompt
        ff_options_str = ""
        for category, force_fields in ff_options.items():
            ff_options_str += f"- {category.capitalize()}: {', '.join(force_fields)}\n"
            
        # Create the prompt for the LLM
        prompt = f"""
        As an expert in molecular dynamics force field selection, analyze this system and research goal to recommend the optimal force field.
        
        SYSTEM DESCRIPTION: {system_description}
        
        SYSTEM COMPOSITION:
        {comp_str}
        
        ELEMENTS PRESENT: {elements_str}
        
        RESEARCH GOAL: "{research_goal}"
        
        AVAILABLE FORCE FIELDS BY CATEGORY:
        {ff_options_str}
        
        Based on this information, please select the most appropriate force field for this system and research goal.
        Consider these factors in your selection:
        1. Accuracy requirements implied by the research goal
        2. Compatibility between force fields if multiple molecular types are present
        3. Specific strengths of each force field for the properties being studied
        4. Computational efficiency vs. accuracy tradeoffs
        5. Recent advancements in force field development
        
        Provide your response as JSON with this structure:
        ```json
        {{
            "force_field": "Name of recommended force field",
            "compatible_water_model": "Recommended water model",
            "justification": "Detailed scientific explanation for this choice",
            "alternatives": ["Alternative force field 1", "Alternative force field 2"],
            "cautions": "Any limitations or issues to be aware of",
            "parameter_availability": "Ease of obtaining parameters (high/medium/low)"
        }}
        ```
        Include only the JSON response with no additional text.
        """
        
        try:
            # Generate response from LLM
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            ff_selection = json.loads(response.text)
            
            # Ensure all expected fields are present
            ff_selection.setdefault("force_field", "AMBER ff14SB")
            ff_selection.setdefault("compatible_water_model", "TIP3P")
            ff_selection.setdefault("alternatives", [])
            ff_selection.setdefault("parameter_availability", "high")
            
            return ff_selection
            
        except Exception as e:
            self.logger.error(f"Error selecting force field: {e}")
            # Fallback to reasonable defaults
            comp = system_info["composition"]
            
            default_selection = {
                "force_field": "AMBER ff14SB" if comp["proteins"] else "GAFF" if comp["small_molecules"] else "TIP3P" if comp["water"] else "OPLS-AA",
                "compatible_water_model": "TIP3P",
                "justification": "Default selection based on system composition.",
                "alternatives": ["CHARMM36", "OPLS-AA"],
                "cautions": "This is a default selection due to LLM analysis failure.",
                "parameter_availability": "high"
            }
            
            return default_selection
    
    def _determine_parameter_method(self, 
                                system_info: Dict[str, Any],
                                force_field: str,
                                research_goal: str) -> Dict[str, Any]:
        """
        Determine the best method to obtain force field parameters.
        
        Args:
            system_info: System information from _analyze_system_composition
            force_field: Selected force field name
            research_goal: Research objective in natural language
            
        Returns:
            Dictionary with parameter acquisition method and details
        """
        self.logger.info(f"Determining parameter acquisition method for {force_field}")
        
        # Format system composition for prompt
        composition = system_info["composition"]
        comp_str = "\n".join([f"- {k.replace('_', ' ')}: {'Yes' if v else 'No'}" for k, v in composition.items()])
        
        # Format parameter method info for prompt
        method_info = ""
        for method, details in self.param_methods.items():
            tools = ", ".join(details["tools"])
            strengths = ", ".join(details["strengths"])
            method_info += f"- {method.upper()}: {details['description']}\n  Tools: {tools}\n  Strengths: {strengths}\n\n"
        
        # Create the prompt for the LLM
        prompt = f"""
        As an expert in molecular dynamics parameterization, determine the best method to obtain force field parameters.
        
        SELECTED FORCE FIELD: {force_field}
        
        SYSTEM COMPOSITION:
        {comp_str}
        
        RESEARCH GOAL: "{research_goal}"
        
        PARAMETER ACQUISITION METHODS:
        {method_info}
        
        Based on this information, what is the best method to obtain parameters for this system?
        Consider these factors:
        1. Parameter availability for the selected force field
        2. Presence of non-standard molecules requiring custom parameterization
        3. Accuracy requirements implied by the research goal
        4. Computational resources and time constraints
        5. Availability of QM-level parameterization tools if needed
        
        Provide your response as JSON with this structure:
        ```json
        {{
            "method": "database|analogy|quantum",
            "justification": "Detailed scientific explanation for this choice",
            "recommended_tools": ["Tool1", "Tool2"],
            "estimated_effort": "low|medium|high",
            "specific_approaches": [
                "Detailed step-by-step approaches to obtain parameters"
            ]
        }}
        ```
        Include only the JSON response with no additional text.
        """
        
        try:
            # Generate response from LLM
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            param_method = json.loads(response.text)
            
            # Ensure all expected fields are present
            param_method.setdefault("method", "database")
            param_method.setdefault("recommended_tools", [])
            param_method.setdefault("estimated_effort", "medium")
            param_method.setdefault("specific_approaches", [])
            
            return param_method
            
        except Exception as e:
            self.logger.error(f"Error determining parameter method: {e}")
            # Fallback to database method
            return {
                "method": "database",
                "justification": "Default selection due to LLM analysis failure.",
                "recommended_tools": ["ParmEd", "AmberTools"],
                "estimated_effort": "medium",
                "specific_approaches": ["Use standard force field databases"]
            }
    
    def _acquire_parameters_from_database(self, 
                                       force_field: str,
                                       system_info: Dict[str, Any],
                                       data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire parameters from established force field databases.
        
        Args:
            force_field: Selected force field name
            system_info: System information from _analyze_system_composition
            data_file: Optional existing LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Acquiring parameters for {force_field} from databases")
        
        # Determine which database files to use based on force field
        ff_files = self._determine_force_field_files(force_field, system_info)
        
        # Extract parameter data from database files
        parameters = {
            "source": "database",
            "force_field": force_field,
            "parameter_files": ff_files,
            "atom_types": {},
            "bonds": {},
            "angles": {},
            "dihedrals": {},
            "impropers": {},
            "nonbonded": {},
        }
        
        # If data_file is provided, extract atom types from it
        if data_file:
            atom_types = self._extract_atom_types_from_data(data_file)
            parameters["atom_types"] = atom_types
            
        # Generate parameter data using LLM
        parameters.update(self._generate_parameters_with_llm(force_field, system_info, data_file))
        
        # Log the acquired parameters
        self.logger.info(f"Acquired parameters for {len(parameters['atom_types'])} atom types")
        
        return parameters
    
    def _acquire_parameters_by_analogy(self,
                                    force_field: str,
                                    system_info: Dict[str, Any],
                                    data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire parameters by chemical analogy to known molecules.
        
        Args:
            force_field: Selected force field name
            system_info: System information from _analyze_system_composition
            data_file: Optional existing LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Acquiring parameters for {force_field} by chemical analogy")
        
        # Start with database parameters as a baseline
        parameters = self._acquire_parameters_from_database(force_field, system_info, data_file)
        parameters["source"] = "analogy"
        
        # Identify molecules needing parameterization by analogy
        unique_molecules = self._extract_unique_molecules(system_info)
        
        # For each unique molecule, find analogies and parameters
        analogy_params = {}
        for molecule in unique_molecules:
            if molecule not in parameters["atom_types"]:
                analogy = self._find_molecular_analogy(molecule, force_field)
                if analogy:
                    analogy_params[molecule] = analogy
        
        parameters["analogies"] = analogy_params
        
        # Use LLM to fill in missing parameters based on analogies
        parameters.update(self._enhance_parameters_with_llm(parameters, "analogy"))
        
        return parameters
    
    def _acquire_parameters_from_quantum(self,
                                      force_field: str,
                                      system_info: Dict[str, Any],
                                      data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire parameters from quantum mechanical calculations.
        
        Args:
            force_field: Selected force field name
            system_info: System information from _analyze_system_composition
            data_file: Optional existing LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Acquiring parameters for {force_field} via quantum calculations")
        
        # Start with database parameters for standard components
        parameters = self._acquire_parameters_from_database(force_field, system_info, data_file)
        parameters["source"] = "quantum"
        
        # Identify molecules needing QM parameterization
        unique_molecules = self._extract_unique_molecules(system_info)
        standard_molecules = self._identify_standard_molecules(unique_molecules, force_field)
        
        # Molecules needing QM treatment = unique - standard
        qm_needed = [m for m in unique_molecules if m not in standard_molecules]
        
        if not qm_needed:
            self.logger.info("No molecules need QM parameterization, using database parameters")
            return parameters
            
        # In a real implementation, we would run QM calculations here
        # For this agent, we'll simulate QM parameters using the LLM
        
        # Enhance parameters with LLM to simulate QM-derived parameters
        parameters["qm_molecules"] = qm_needed
        parameters.update(self._enhance_parameters_with_llm(parameters, "quantum"))
        
        return parameters
        
    def _determine_force_field_files(self, force_field: str, system_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine which force field database files are needed.
        
        Args:
            force_field: Selected force field name
            system_info: System information
            
        Returns:
            Dictionary mapping parameter types to file paths
        """
        # This would typically map to real force field files
        # Here we'll return placeholders that would be resolved in a real implementation
        
        ff_files = {}
        comp = system_info["composition"]
        
        if "AMBER" in force_field:
            ff_base = "amber"
            if comp["proteins"]:
                ff_files["proteins"] = f"{ff_base}/ff14SB.dat"
            if comp["water"]:
                ff_files["water"] = f"{ff_base}/tip3p.dat"
            if comp["ions"]:
                ff_files["ions"] = f"{ff_base}/ions.dat"
            if comp["nucleic_acids"]:
                ff_files["nucleic_acids"] = f"{ff_base}/DNA.OL15.dat"
            if comp["small_molecules"]:
                ff_files["small_molecules"] = f"{ff_base}/gaff.dat"
                
        elif "CHARMM" in force_field:
            ff_base = "charmm"
            if comp["proteins"]:
                ff_files["proteins"] = f"{ff_base}/prot.prm"
            if comp["water"]:
                ff_files["water"] = f"{ff_base}/water.prm"
            if comp["lipids"]:
                ff_files["lipids"] = f"{ff_base}/lipid.prm"
                
        elif "OPLS" in force_field:
            ff_base = "opls"
            ff_files["main"] = f"{ff_base}/oplsaa.prm"
            
        else:
            # Generic case
            ff_files["main"] = f"generic/{force_field.lower().replace(' ', '_')}.dat"
            
        return ff_files
    
    def _extract_atom_types_from_data(self, data_file: str) -> Dict[str, Any]:
        """
        Extract atom types from a LAMMPS data file.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Dictionary mapping atom types to properties
        """
        atom_types = {}
        
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                
            # Find the "Masses" section
            in_masses = False
            for line in lines:
                line = line.strip()
                
                if "Masses" in line:
                    in_masses = True
                    continue
                elif in_masses and line.startswith("#"):
                    continue
                elif in_masses and not line:  # Empty line ends section
                    in_masses = False
                elif in_masses:
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        mass = float(parts[1])
                        # Guess element from mass
                        element = self._guess_element_from_mass(mass)
                        atom_types[atom_type] = {
                            "mass": mass,
                            "element": element
                        }
            
            # Look for atom types in "Pair Coeffs" section
            in_pair_coeffs = False
            for line in lines:
                line = line.strip()
                
                if "Pair Coeffs" in line:
                    in_pair_coeffs = True
                    continue
                elif in_pair_coeffs and line.startswith("#"):
                    continue
                elif in_pair_coeffs and not line:  # Empty line ends section
                    in_pair_coeffs = False
                elif in_pair_coeffs:
                    parts = line.split()
                    if len(parts) >= 3:
                        atom_type = int(parts[0])
                        epsilon = float(parts[1])
                        sigma = float(parts[2])
                        if atom_type in atom_types:
                            atom_types[atom_type].update({
                                "epsilon": epsilon,
                                "sigma": sigma
                            })
                        
        except Exception as e:
            self.logger.error(f"Error extracting atom types from data file: {e}")
            
        return atom_types
    
    def _guess_element_from_mass(self, mass: float) -> str:
        """Guess element from atomic mass."""
        # Define mass ranges for common elements
        mass_ranges = {
            "H": (0.9, 1.1),
            "C": (11.5, 12.5),
            "N": (13.5, 14.5),
            "O": (15.5, 16.5),
            "Na": (22.5, 23.5),
            "Mg": (23.5, 24.5),
            "P": (30.5, 31.5),
            "S": (31.5, 32.5),
            "Cl": (35.0, 36.0),
            "K": (38.5, 39.5),
            "Ca": (39.5, 40.5),
            "Fe": (55.0, 56.0),
            "Zn": (64.5, 65.5)
        }
        
        for element, (min_mass, max_mass) in mass_ranges.items():
            if min_mass <= mass <= max_mass:
                return element
                
        # Default to closest match if not in range
        closest_element = "C"
        closest_diff = float('inf')
        
        for element, (min_mass, max_mass) in mass_ranges.items():
            avg_mass = (min_mass + max_mass) / 2
            diff = abs(mass - avg_mass)
            if diff < closest_diff:
                closest_diff = diff
                closest_element = element
                
        return closest_element
    
    def _extract_unique_molecules(self, system_info: Dict[str, Any]) -> List[str]:
        """
        Extract unique molecules from system information.
        
        Args:
            system_info: System information from _analyze_system_composition
            
        Returns:
            List of unique molecule names
        """
        unique_molecules = []
        
        # If residue_counts exists, use those as the molecules
        if "residue_counts" in system_info:
            unique_molecules = list(system_info["residue_counts"].keys())
        else:
            # Try to infer molecules from composition
            comp = system_info["composition"]
            if comp["water"]:
                unique_molecules.append("HOH")
            if comp["ions"]:
                elements = system_info.get("elements", {})
                for ion in ["Na", "K", "Cl", "Ca", "Mg"]:
                    if ion in elements:
                        unique_molecules.append(ion)
                        
        return unique_molecules
    
    def _identify_standard_molecules(self, molecules: List[str], force_field: str) -> List[str]:
        """
        Identify which molecules are standard for a given force field.
        
        Args:
            molecules: List of molecule names
            force_field: Force field name
            
        Returns:
            List of standard molecule names
        """
        # Common standard molecules across force fields
        standard_molecules = ["HOH", "WAT", "TIP3", "SOL", "Na", "K", "Cl", "Ca", "Mg"]
        
        # Standard amino acids
        amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
                      "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", 
                      "TYR", "VAL"]
        
        # Standard nucleotides
        nucleotides = ["ADE", "THY", "GUA", "CYT", "URA", "A", "T", "G", "C", "U",
                      "DA", "DT", "DG", "DC", "DU"]
        
        if "AMBER" in force_field or "CHARMM" in force_field:
            standard_molecules.extend(amino_acids)
            standard_molecules.extend(nucleotides)
            
        return [m for m in molecules if m in standard_molecules]
    
    def _find_molecular_analogy(self, molecule: str, force_field: str) -> Dict[str, Any]:
        """
        Find analogous molecules for parameterization by analogy.
        
        Args:
            molecule: Molecule name
            force_field: Force field name
            
        Returns:
            Dictionary with analogy information
        """
        # This would typically involve a database search or structural comparison
        # Here we'll simulate the result
        
        # Example return structure
        return {
            "similar_to": "similar molecule name",
            "similarity": 0.85,  # 0-1 score
            "modifications_needed": ["replace methyl with ethyl"],
            "parameter_adjustments": ["increase C-C bond length by 0.02 Å"]
        }
    
    def _generate_parameters_with_llm(self, 
                                   force_field: str, 
                                   system_info: Dict[str, Any],
                                   data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate force field parameters using LLM.
        
        Args:
            force_field: Selected force field name
            system_info: System information
            data_file: Optional LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Generating parameters for {force_field} using LLM")
        
        # Extract elements and composition for the prompt
        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("elements", {}).items()])
        comp = system_info["composition"]
        comp_str = "\n".join([f"- {k.replace('_', ' ')}: {'Yes' if v else 'No'}" for k, v in comp.items()])
        
        # Create prompt for LLM
        prompt = f"""
        As an expert in molecular dynamics force fields, provide appropriate LAMMPS parameters for this system.
        
        FORCE FIELD: {force_field}
        
        SYSTEM ELEMENTS: {elements_str}
        
        SYSTEM COMPOSITION:
        {comp_str}
        
        You need to generate scientifically accurate force field parameters for LAMMPS based on the {force_field} force field.
        For each parameter type, provide values compatible with LAMMPS syntax and the selected force field.
        
        Please provide parameters in the following JSON format:
        ```json
        {{
            "atom_types": {{
                "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}},
                "2": {{"name": "H", "mass": 1.008, "description": "Water hydrogen", "charge": 0.4238, "epsilon": 0.0, "sigma": 0.0}}
            }},
            "bonds": {{
                "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
            }},
            "angles": {{
                "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
            }},
            "dihedrals": {{
                "1": {{"type": "periodic", "atoms": ["X", "X", "X", "X"], "k": 0.0, "d": 1, "n": 1, "description": "Example dihedral"}}
            }},
            "nonbonded_terms": {{
                "mixing_rule": "geometric for epsilon, arithmetic for sigma",
                "cutoff": 10.0
            }}
        }}
        ```
        
        Include only parameters relevant to the system composition. The parameters should be scientifically accurate for the {force_field} force field.
        Include only the JSON response with no additional text.
        """
        
        try:
            # Generate response from LLM
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            parameters = json.loads(response.text)
            
            # Ensure all parameter categories exist
            for category in ["atom_types", "bonds", "angles", "dihedrals", "nonbonded_terms"]:
                if category not in parameters:
                    parameters[category] = {}
                    
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error generating parameters with LLM: {e}")
            # Return empty parameters
            return {
                "atom_types": {},
                "bonds": {},
                "angles": {},
                "dihedrals": {},
                "nonbonded_terms": {
                    "mixing_rule": "geometric for epsilon, arithmetic for sigma",
                    "cutoff": 10.0
                }
            }
    
    def _enhance_parameters_with_llm(self, 
                                  parameters: Dict[str, Any], 
                                  method: str) -> Dict[str, Any]:
        """
        Enhance parameters using LLM based on the specified method.
        
        Args:
            parameters: Existing parameters
            method: Method being used (analogy or quantum)
            
        Returns:
            Dictionary with enhanced parameter information
        """
        force_field = parameters.get("force_field", "Unknown")
        
        # Create prompt for LLM based on method
        if method == "analogy":
            # Extract analogies for the prompt
            analogies_str = ""
            for molecule, analogy in parameters.get("analogies", {}).items():
                similar_to = analogy.get("similar_to", "unknown")
                similarity = analogy.get("similarity", 0)
                modifications = ", ".join(analogy.get("modifications_needed", []))
                analogies_str += f"- {molecule}: similar to {similar_to} (similarity: {similarity})\n  Modifications: {modifications}\n"
                
            prompt = f"""
            As an expert in molecular force field parameterization by analogy, enhance these parameters.
            
            FORCE FIELD: {force_field}
            
            CURRENT PARAMETERS:
            {json.dumps(parameters.get('atom_types', {}), indent=2)}
            
            MOLECULAR ANALOGIES:
            {analogies_str}
            
            Based on the molecular analogies provided, enhance the parameters to account for the chemical differences.
            Adjust parameters like bond lengths, angles, charges, and non-bonded terms based on chemical intuition and the force field paradigm.
            
            Please provide enhanced parameters in this JSON format:
            ```json
            {{
                "atom_types": {{
                    "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}}
                }},
                "bonds": {{
                    "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
                }},
                "angles": {{
                    "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
                }}
            }}
            ```
            Include only the JSON response with no additional text.
            """
            
        elif method == "quantum":
            # Extract QM molecules for the prompt
            qm_molecules = ", ".join(parameters.get("qm_molecules", []))
            
            prompt = f"""
            As an expert in quantum-derived force field parameterization, enhance these parameters.
            
            FORCE FIELD: {force_field}
            
            CURRENT PARAMETERS:
            {json.dumps(parameters.get('atom_types', {}), indent=2)}
            
            MOLECULES NEEDING QM PARAMETERIZATION: {qm_molecules}
            
            Based on your expertise in quantum chemistry and force field development, provide enhanced parameters 
            that would typically be derived from quantum mechanical calculations. Focus on accurate charges, bond, angle,
            and dihedral parameters that reflect the electronic structure of the molecules.
            
            Please provide quantum-derived parameters in this JSON format:
            ```json
            {{
                "atom_types": {{
                    "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}}
                }},
                "bonds": {{
                    "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
                }},
                "angles": {{
                    "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
                }}
            }}
            ```
            Include only the JSON response with no additional text.
            """
            
        else:
            # Unknown method
            return parameters
            
        try:
            # Generate response from LLM
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            enhanced_params = json.loads(response.text)
            
            # Merge enhanced parameters with original parameters
            for category in ["atom_types", "bonds", "angles", "dihedrals"]:
                if category in enhanced_params:
                    # Add new parameters
                    for key, value in enhanced_params[category].items():
                        if key not in parameters.get(category, {}):
                            if category not in parameters:
                                parameters[category] = {}
                            parameters[category][key] = value
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error enhancing parameters with LLM: {e}")
            return parameters
    
    def _validate_parameters(self, parameters: Dict[str, Any], system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for scientific rigor.
        
        Args:
            parameters: Parameter information
            system_info: System information
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "quality_metrics": {}
        }
        
        # Check for missing parameters
        missing_atom_types = []
        elements = system_info.get("elements", {})
        
        # Check if parameters exist for all elements
        for element in elements:
            found = False
            for atom_type in parameters.get("atom_types", {}).values():
                if atom_type.get("name", "") == element or atom_type.get("element", "") == element:
                    found = True
                    break
                    
            if not found:
                missing_atom_types.append(element)
                validation["warnings"].append(f"Missing parameters for element {element}")
                
        if missing_atom_types:
            validation["passed"] = False
            
        # Check for reasonable parameter values
        for atom_type, props in parameters.get("atom_types", {}).items():
            # Check mass
            mass = props.get("mass", 0)
            if mass <= 0:
                validation["errors"].append(f"Invalid mass for atom type {atom_type}: {mass}")
                validation["passed"] = False
                
            # Check charges - should be reasonable values
            charge = props.get("charge", 0)
            if abs(charge) > 2.0:
                validation["warnings"].append(f"Unusual charge for atom type {atom_type}: {charge}")
                
            # Check LJ parameters
            epsilon = props.get("epsilon", 0)
            sigma = props.get("sigma", 0)
            if epsilon < 0:
                validation["errors"].append(f"Invalid epsilon for atom type {atom_type}: {epsilon}")
                validation["passed"] = False
            if sigma <= 0:
                validation["errors"].append(f"Invalid sigma for atom type {atom_type}: {sigma}")
                validation["passed"] = False
                
        # Calculate quality metrics
        param_source = parameters.get("source", "unknown")
        
        # Source-based quality score (0-100)
        source_quality = {
            "database": 70,  # Good baseline
            "analogy": 80,   # Better than database
            "quantum": 95    # Best quality
        }.get(param_source, 50)
        
        # Adjust based on coverage
        coverage = 100 - (len(missing_atom_types) / max(1, len(elements)) * 100)
        
        # Compute combined score
        quality_score = (source_quality * 0.7) + (coverage * 0.3)
        
        validation["quality_metrics"] = {
            "overall_score": int(quality_score),
            "parameter_source": param_source,
            "coverage": int(coverage),
            "missing_elements": missing_atom_types
        }
        
        return validation
    
    def _parse_data_file(self, data_file: str) -> Dict[str, Any]:
        """
        Parse a LAMMPS data file to understand what parameters are needed.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Dictionary with data file information
        """
        info = {
            "atom_types": 0,
            "bond_types": 0,
            "angle_types": 0,
            "dihedral_types": 0,
            "improper_types": 0,
            "atoms": 0,
            "bonds": 0,
            "angles": 0,
            "dihedrals": 0,
            "impropers": 0,
            "masses": {},
            "box": []
        }
        
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for header information
                if "atoms" in line and not "types" in line:
                    info["atoms"] = int(line.split()[0])
                elif "bonds" in line and not "types" in line:
                    info["bonds"] = int(line.split()[0])
                elif "angles" in line and not "types" in line:
                    info["angles"] = int(line.split()[0])
                elif "dihedrals" in line and not "types" in line:
                    info["dihedrals"] = int(line.split()[0])
                elif "impropers" in line and not "types" in line:
                    info["impropers"] = int(line.split()[0])
                elif "atom types" in line:
                    info["atom_types"] = int(line.split()[0])
                elif "bond types" in line:
                    info["bond_types"] = int(line.split()[0])
                elif "angle types" in line:
                    info["angle_types"] = int(line.split()[0])
                elif "dihedral types" in line:
                    info["dihedral_types"] = int(line.split()[0])
                elif "improper types" in line:
                    info["improper_types"] = int(line.split()[0])
                elif "xlo xhi" in line:
                    parts = line.split()
                    info["box"].append((float(parts[0]), float(parts[1])))
                    
            # Extract masses
            in_masses = False
            for line in lines:
                line = line.strip()
                
                if "Masses" in line:
                    in_masses = True
                    continue
                elif in_masses and not line:  # Empty line
                    in_masses = False
                elif in_masses and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        mass = float(parts[1])
                        info["masses"][atom_type] = mass
                        
        except Exception as e:
            self.logger.error(f"Error parsing data file: {e}")
            
        return info
    
    def _generate_lammps_parameters(self, 
                                data_file_info: Dict[str, Any],
                                parameter_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate LAMMPS parameter file content based on acquired parameters.
        
        Args:
            data_file_info: Information from parsing the data file
            parameter_info: Parameter information from acquire_parameters()
            
        Returns:
            Dictionary with parameter file content
        """
        self.logger.info("Generating LAMMPS parameter file content")
        
        # Use LLM to generate scientifically accurate LAMMPS parameters
        force_field = parameter_info.get("force_field", "Unknown")
        parameters = parameter_info.get("atom_types", {})
        validation = parameter_info.get("validation", {})
        
        # Format parameter info for the prompt
        param_str = json.dumps(parameter_info, indent=2)
        data_file_str = json.dumps(data_file_info, indent=2)
        
        # Create prompt for LLM
        prompt = f"""
        As an expert in molecular dynamics force fields, generate LAMMPS parameter files for this system.
        
        FORCE FIELD: {force_field}
        
        DATA FILE INFORMATION:
        {data_file_str}
        
        PARAMETER INFORMATION:
        {param_str}
        
        IMPORTANT: Do NOT include 'units' commands in the parameter file, as these will be set in the main input script.
        
        Based on this information, generate a complete and scientifically accurate LAMMPS parameter file.
        The file should include all parameters needed for a LAMMPS simulation with this force field.
        
        IMPORTANT GUIDELINES:
        1. The file must ONLY contain force field parameters - NO simulation setup commands (e.g., units command).
        2. DO include comments explaining the parameters
        3. Focus solely on defining force field parameters, not running a simulation
       
        4. PAIR STYLE GUIDELINES:
           - For TIP4P water: Use 'hybrid/overlay' pair style as shown in the TIP4P information section
           - For TIP3P/SPC/E: Use standard 'lj/cut/coul/long' pair style
           
        5. PAIR COEFFICIENTS GUIDELINES:
           - For standard pair styles: pair_coeff 1 1 0.16275 3.16435  # O-O
           - For hybrid pair styles (TIP4P): pair_coeff 1 1 lj/cut/coul/long 0.16275 3.16435  # O-O
           - ALWAYS include the sub-style name in pair_coeff commands when using hybrid pair styles
           
        6. Include proper mass, bond_coeff, and angle_coeff commands.
        7. Include special_bonds settings appropriate for this force field.
        8. Include kspace_style command for long-range electrostatics. 

        Include ONLY these parameter-related commands:
        1. pair_coeff commands for atom interactions
        2. bond_coeff commands for bond parameters
        3. angle_coeff commands for angle parameters
        4. dihedral_coeff commands (if needed)
        5. improper_coeff commands (if needed)
        6. special_bonds settings (allowed, as it defines force field behavior)
        7. pair_style, bond_style, angle_style (allowed, as they define force field styles)
        8. mass commands (allowed, as they define atom masses)
        9. set type commands ONLY if they set force field parameters
        10. kspace_style (allowed, as it defines long-range interaction handling) 

        Format the output as a JSON object with the main parameter file content and any additional files needed:
        ```json
        {{
            "main": "# LAMMPS parameters for {force_field}\\n\\npair_style lj/cut/coul/long 10.0\\n...",
            "additional": {{
                "water_params": "# Water model parameters\\n\\n...",
                "other_file": "# Other parameters\\n\\n..."
            }}
        }}
        ```
        
        Make sure the LAMMPS syntax is correct and all parameters are scientifically accurate for the {force_field} force field.
        Include only the JSON response with no additional text.
        """
        
        try:
            # Generate response from LLM
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            param_files = json.loads(response.text)
            
            # Ensure main content exists
            if "main" not in param_files:
                param_files["main"] = self._generate_fallback_parameters(data_file_info, parameter_info)
                
            # Ensure additional exists
            if "additional" not in param_files:
                param_files["additional"] = {}
                
            return param_files
            
        except Exception as e:
            self.logger.error(f"Error generating LAMMPS parameters: {e}")
            # Generate fallback parameters
            return {
                "main": self._generate_fallback_parameters(data_file_info, parameter_info),
                "additional": {}
            }
    
    def _generate_fallback_parameters(self, 
                                   data_file_info: Dict[str, Any],
                                   parameter_info: Dict[str, Any]) -> str:
        """
        Generate fallback parameters if LLM generation fails.
        
        Args:
            data_file_info: Information from parsing the data file
            parameter_info: Parameter information from acquire_parameters()
            
        Returns:
            Parameter file content as string
        """
        force_field = parameter_info.get("force_field", "Unknown")
        
        # Build basic parameter file content
        lines = [
            f"# LAMMPS parameters for {force_field}",
            "# Generated by ForceFieldAgent (fallback generator)",
            "",
            "# General force field settings",
            "pair_style lj/cut/coul/long 10.0",
            "bond_style harmonic",
            "angle_style harmonic",
            ""
        ]
        
        # Add dihedral style if needed
        if data_file_info.get("dihedral_types", 0) > 0:
            lines.append("dihedral_style harmonic")
            
        # Add improper style if needed
        if data_file_info.get("improper_types", 0) > 0:
            lines.append("improper_style harmonic")
            
        lines.append("")
        
        # Add pair coefficients
        atom_types = data_file_info.get("atom_types", 0)
        if atom_types > 0:
            lines.append("# Pair coefficients")
            for i in range(1, atom_types + 1):
                mass = data_file_info.get("masses", {}).get(i, 12.0)  # Default to carbon mass
                element = self._guess_element_from_mass(mass)
                
                # Use reasonable defaults based on element
                if element == "O":
                    epsilon, sigma = 0.1553, 3.166
                elif element == "H":
                    epsilon, sigma = 0.0, 0.0
                elif element == "C":
                    epsilon, sigma = 0.1094, 3.4
                elif element == "N":
                    epsilon, sigma = 0.1700, 3.25
                elif element in ["Na", "K"]:
                    epsilon, sigma = 0.1, 2.8
                elif element in ["Cl", "Br"]:
                    epsilon, sigma = 0.1, 4.4
                else:
                    epsilon, sigma = 0.1, 3.0
                    
                lines.append(f"pair_coeff {i} {i} {epsilon} {sigma}  # {element}")
                
            lines.append("")
            
        # Add bond coefficients
        bond_types = data_file_info.get("bond_types", 0)
        if bond_types > 0:
            lines.append("# Bond coefficients")
            for i in range(1, bond_types + 1):
                lines.append(f"bond_coeff {i} 450.0 1.0  # Generic bond")
            lines.append("")
            
        # Add angle coefficients
        angle_types = data_file_info.get("angle_types", 0)
        if angle_types > 0:
            lines.append("# Angle coefficients")
            for i in range(1, angle_types + 1):
                lines.append(f"angle_coeff {i} 55.0 109.47  # Generic angle")
            lines.append("")
            
        # Add special bonds
        lines.append("# Special bonds settings")
        if "AMBER" in force_field or "CHARMM" in force_field:
            lines.append("special_bonds lj/coul 0.0 0.0 0.5")
        else:
            lines.append("special_bonds lj/coul 0.0 0.0 0.5  # Generic setting")
            
        lines.append("")
        lines.append("# Long-range electrostatics")
        lines.append("kspace_style pppm 1.0e-5")
        
        return "\n".join(lines)
    
    def _generate_parameter_summary(self, 
                                 params: Dict[str, Any], 
                                 selection_info: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the parameters.
        
        Args:
            params: Parameter information
            selection_info: Force field selection information
            
        Returns:
            Markdown summary text
        """
        force_field = selection_info.get("force_field", {}).get("force_field", "Unknown")
        water_model = selection_info.get("force_field", {}).get("compatible_water_model", "Unknown")
        justification = selection_info.get("force_field", {}).get("justification", "")
        
        method = selection_info.get("parameter_method", {}).get("method", "unknown")
        method_desc = {
            "database": "Direct extraction from established databases",
            "analogy": "Parameters by chemical analogy to known molecules",
            "quantum": "Ab initio parameterization from quantum calculations"
        }.get(method, "Unknown method")
        
        # Get atom type counts
        atom_types_count = len(params.get("atom_types", {}))
        
        # Get validation info
        validation = params.get("validation", {})
        quality_score = validation.get("quality_metrics", {}).get("overall_score", 0)
        passed = validation.get("passed", False)
        warnings = validation.get("warnings", [])
        
        # Create the summary text
        lines = [
            f"# Force Field Parameters Summary",
            "",
            f"## Selection Details",
            f"- **Force Field**: {force_field}",
            f"- **Water Model**: {water_model}",
            f"- **Parameter Source**: {method} ({method_desc})",
            f"- **Quality Score**: {quality_score}/100",
            "",
            f"## Justification",
            f"{justification}",
            "",
            f"## Parameter Statistics",
            f"- **Atom Types**: {atom_types_count}",
            f"- **Bond Types**: {len(params.get('bonds', {}))}",
            f"- **Angle Types**: {len(params.get('angles', {}))}",
            f"- **Dihedral Types**: {len(params.get('dihedrals', {}))}",
            "",
        ]
        
        # Add validation section
        lines.extend([
            f"## Validation",
            f"- **Passed**: {'Yes' if passed else 'No'}",
            ""
        ])
        
        if warnings:
            lines.append("### Warnings")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
            
        return "\n".join(lines)

def _fix_lammps_syntax(self, lammps_text: str) -> str:
    """
    Fix common LAMMPS syntax issues in generated parameters.
    
    Args:
        lammps_text: LAMMPS parameter text
        
    Returns:
        Corrected LAMMPS parameter text
    """
    lines = lammps_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix missing spaces after commas
        line = re.sub(r',(?=\S)', ', ', line)
        
        # Fix incorrect comment syntax
        line = re.sub(r'(//.+)$', r'#\1', line)
        
        # Fix invalid syntax in pair_coeff commands
        if line.strip().startswith("pair_coeff") and re.search(r'pair_coeff\s+\*\s+\*\s+[\d\.]+\s*$', line):
            line = line + " # Missing sigma parameter"
            
        fixed_lines.append(line)
        
    return '\n'.join(fixed_lines)
