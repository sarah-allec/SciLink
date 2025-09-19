import os
import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from .instruct import MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS


class MoltemplateAgent:
    """
    A generalized agent for generating LAMMPS input files via Moltemplate.
    Dynamically handles force fields, PDB updates, template generation, and system.lt creation.
    """

    def __init__(self, working_dir: str, ff_library_path: str, description: str, system_pdb: str, packmol_input: str,
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-pro-preview-05-06"):
        """
        Initializes the MoltemplateAgent.

        :param working_dir: Directory for intermediate and output file processing.
        :param ff_library_path: Directory path containing force field `.lt` files.
        :param api_key: API key for Generative AI.
        :param model_name: The Generative AI model to utilize.
        """
        self.working_dir = working_dir
        self.ff_library_path = ff_library_path
        self.description = description
        self.system_pdb = system_pdb
        self.packmol_input = packmol_input
        os.makedirs(working_dir, exist_ok=True)

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Configure Generative AI
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("An API key must be provided or set via GOOGLE_API_KEY.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")

    # ============
    # PUBLIC API
    # ============
    def generate_lammps_input(self) -> Dict[str, str]:
        """
        Full workflow: PDB updates, dynamic force field handling, and system.lt generation.

        :param description: High-level description of the system (e.g., "0.5M NaCl in water").
        :param system_pdb: Path to the input PDB (from PACKMOL or similar tools).
        :param packmol_input: Path to PACKMOL input file (optional, additional metadata for molecules).
        :return: Dictionary with generation status and file paths.
        """
        try:
            # Step 1: Parse and select force fields dynamically
            force_fields = self._parse_force_fields() # return dictionary of file name and template

            # Step 2: Reorder the PDB file
            updated_pdb_path = self._rename_molecules() # only reorder here

            # Step 3: Count molecules dynamically from updated PDB
            molecule_counts = self._count_molecules(updated_pdb_path)

            # Step 4: Generate system.lt dynamically using Generative AI
            system_lt_path = self._generate_system_lt_with_llm(
                self.description, molecule_counts, force_fields 
            )


            return {
                "status": "success",
                "working_dir": self.working_dir,
                "updated_pdb": updated_pdb_path,
                "system_lt": system_lt_path,
            }
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    # ====================
    # FORCE FIELD PARSING
    # ====================
    def _parse_force_fields(self) -> Dict[str, List[str]]:
        """
        Parses `.lt` files to detect molecule templates dynamically.

        :return: A dictionary mapping filenames to the defined molecule templates.
        """
        self.logger.info("Parsing `.lt` files to detect molecule templates...")
        force_fields = {}
        for ff_file in os.listdir(self.ff_library_path):
            if ff_file.endswith(".lt"):
                full_path = os.path.join(self.ff_library_path, ff_file)
                templates = self._discover_templates_in_lt(full_path)
                force_fields[ff_file] = templates
        self.logger.info(f"Parsed force field files: {force_fields}")
        return force_fields

    def _discover_templates_in_lt(self, lt_file: str) -> List[str]:
        """
        Parses a `.lt` force field file to detect defined molecule templates using `inherits`.

        :param lt_file: Path to an `.lt` file.
        :return: A list of molecule templates in the file.
        """
        templates = []
        try:
            with open(lt_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):  # Skip blank/commented lines
                        match = re.match(r'^(\w+)\s+inherits\s+.*$', line)
                        if match:
                            templates.append(match.group(1))  # Extract template name
        except Exception as e:
            self.logger.warning(f"Error parsing `{lt_file}`: {e}")
        return templates

    # ==================
    # PDB PROCESSING
    # ==================
    def _rename_molecules(self, output_pdb_file: str = None) -> None:
        """
        Update residue names in a PDB file based on the Packmol input file.
    
        Args:
            pdb_file (str): Path to the input PDB file.
            packmol_input_file (str): Path to the Packmol input file.
            output_pdb_file (str): Path to the output updated PDB file. If None, the output file 
                will be saved as 'renamed_<filename>' in self.working_dir.
        """
        # If output_pdb_file is None, generate it based on pdb_file and save to self.working_dir
        if output_pdb_file is None:
            # Extract the filename from the pdb_file path (e.g., 'nacl_water_0_5M_40A.pdb')
            filename = os.path.basename(self.system_pdb)
            # Prepend 'renamed_' to the filename (e.g., 'renamed_nacl_water_0_5M_40A.pdb')
            renamed_filename = f'renamed_{filename}'
            # Combine with self.working_dir to get the final path
            output_pdb_file = os.path.join(self.working_dir, renamed_filename)
    
        # Parse the Packmol input file to identify the residue mapping
        mappings = {}
        with open(self.packmol_input, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Look for lines starting with `structure`
                # Example: structure components/nacl.pdb
                if line.strip().startswith("structure"):
                    # Extract the file path and molecule name
                    match = re.search(r'structure\s+(\S+)', line)
                    if match:
                        file_path = match.group(1)
                        molecule_name = file_path.split("/")[-1].split(".")[0]  # Extract base name (e.g., nacl)
                        shortened_name = molecule_name[:3].upper()  # Convert to uppercase and shorten to 3 characters
                        mappings[len(mappings) + 1] = shortened_name  # Map molecule index (e.g., MOL A -> NAC)
    
        # Read the input PDB file
        with open(self.system_pdb, "r") as f:
            pdb_lines = f.readlines()
    
        # Process each ATOM line in the PDB file
        updated_pdb_lines = []
        for line in pdb_lines:
            # Only modify lines starting with "ATOM"
            if line.startswith("ATOM"):
                residue_name = line[17:20].strip()  # Extract residue name (e.g., MOL X format)
                # Match residue name (e.g., "MOL A" -> molecule index 1 -> NAC)
                if "MOL" in residue_name:
                    chain_id = line[21].strip()  # Extract chain identifier (e.g., "A" from "MOL A")
                    molecule_index = ord(chain_id) - ord('A') + 1  # Convert chain letter (A -> 1, B -> 2)
                    if molecule_index in mappings:
                        new_residue_name = mappings[molecule_index]
                        # Replace the residue name (fixed-width from position 17 to 20 in PDB format)
                        line = line[:17] + f"{new_residue_name:<3}" + line[20:]
            # Add the line to updated PDB
            updated_pdb_lines.append(line)
    
        # Write the updated PDB file
        with open(output_pdb_file, "w") as f:
            f.writelines(updated_pdb_lines)
    
        print(f"Updated PDB file written to: {output_pdb_file}")
    
        return output_pdb_file

    # ========================
    # SYSTEM CONFIGURATION
    # ========================
    def _count_molecules(self, pdb_path: str) -> Dict[str, int]:
        """ 
        Counts unique molecules from the PDB file by tracking residue name and ID combinations.
        :param pdb_path: Path to the PDB file.
        :return: Dictionary mapping residue names to molecule counts.
        """ 
        self.logger.info("Counting molecules in the updated PDB...")
        molecule_ids = set()  # Set of (residue_name, residue_id) tuples
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f: 
                    if line.startswith(("ATOM", "HETATM")):
                        residue_name = line[17:20].strip()
                        residue_id = line[22:26].strip()  # Residue sequence number
                        molecule_ids.add((residue_name, residue_id))
            
            # Count molecules of each type
            counts = {}
            for res_name, res_id in molecule_ids:
                counts[res_name] = counts.get(res_name, 0) + 1
                
            self.logger.info(f"Molecule counts: {counts}")
            return counts
        except Exception as e:
            self.logger.error(f"Error counting molecules: {e}")
            raise

    def _combine_molecule_count_with_packmol_input(self, molecule_counts: Dict[str, int], packmol_input: str) -> Dict[str, int]:
        """
        Reconciles molecule counts from PACKMOL input and PDB-derived counts.

        :param molecule_counts: Counts derived from the PDB file.
        :param packmol_input: Path to PACKMOL input.
        :return: Updated molecule counts.
        """
        self.logger.info(f"Parsing PACKMOL input: {packmol_input}")
        try:
            with open(packmol_input, 'r') as f:
                for line in f:
                    if line.strip().lower().startswith("structure"):
                        tokens = line.split()
                        if len(tokens) >= 2:
                            molecule_name = tokens[1].split("/")[-1].split(".")[0]
                            molecule_counts[molecule_name] = molecule_counts.get(molecule_name, 0) + 1
            self.logger.info(f"Updated molecule counts with PACKMOL input: {molecule_counts}")
            return molecule_counts
        except Exception as e:
            self.logger.warning(f"Error parsing PACKMOL input: {e}")
            return molecule_counts

    # =============================
    # SYSTEM.LT GENERATION WITH LLM
    # =============================
    def _generate_system_lt_with_llm(self, description: str, molecule_counts: Dict[str, int],
                                   force_fields: Dict[str, List[str]]) -> str:
        """
        Intelligently generates `system.lt` with precise PDB name matching.
        :param description: Description of the system.
        :param molecule_counts: Molecule counts from PDB analysis.
        :param force_fields: Dictionary mapping force field filenames to their templates.
        :return: Path to the generated `system.lt` file.
        """
        self.logger.info("Analyzing system components and generating system.lt...")
        
        # Examine the PDB file to understand molecule structure more deeply
        pdb_molecules = self._analyze_pdb_molecules()
        
        # Format the force field information for the prompt
        force_field_info = []
        for ff_file, templates in force_fields.items():
            templates_str = ", ".join(templates)
            force_field_info.append(f"- {ff_file}: Contains templates: {templates_str}")
        force_field_info_str = "\n".join(force_field_info)
        
        # Format molecule counts as a readable text
        molecule_counts_text = []
        for molecule_type, count in molecule_counts.items():
            molecule_info = f"- {molecule_type}: {count} molecules"
            # Add molecule structure info if available
            if molecule_type in pdb_molecules:
                atoms = pdb_molecules[molecule_type]
                molecule_info += f" (Atoms: {', '.join(atoms)})"
            molecule_counts_text.append(molecule_info)
        molecule_counts_str = "\n".join(molecule_counts_text)
        
        # Format instruction prompt
        prompt = MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS.format(
            system_description=description,
            force_field_info=force_field_info_str,
            molecule_counts=molecule_counts_str
        )
        
        try:
            # Use Generative AI for content generation
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            response_json = json.loads(response.text)
            
            # Log the thought process for debugging
            self.logger.info(f"LLM Analysis: {response_json.get('thought', 'No analysis provided')}")
            
            # Extract the system.lt content and molecule mappings
            lt_content = response_json["system_lt"]
            molecule_mappings = response_json.get("molecule_mappings", {})
            
            # Validate the molecule mappings against the PDB
            if molecule_mappings:
                self.logger.warning(f"Model requested PDB molecule name mappings: {molecule_mappings}")
                self.logger.warning("This indicates a mismatch between system.lt and PDB molecule names.")
                self._update_pdb_molecule_names(molecule_mappings)
            else:
                self.logger.info("No PDB molecule name mappings needed - system.lt uses matching names.")
            
            # Write generated system.lt
            output_path = os.path.join(self.working_dir, "system.lt")
            with open(output_path, 'w') as f:
                f.write(lt_content)
                
            self.logger.info(f"Generated system.lt written to `{output_path}`.")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to generate system.lt via LLM: {e}")
            raise RuntimeError(f"Error generating system.lt: {e}")
    
    def _analyze_pdb_molecules(self) -> Dict[str, List[str]]:
        """
        Analyzes the PDB file to extract atom types for each molecule.
        :return: Dictionary mapping molecule types to their constituent atoms.
        """
        self.logger.info("Analyzing PDB molecule structure...")
        
        # Find the renamed PDB file
        pdb_files = [f for f in os.listdir(self.working_dir) if f.endswith('.pdb')]
        if not pdb_files:
            self.logger.warning("No PDB file found in working directory, using original PDB")
            pdb_path = self.system_pdb
        else:
            # Use the most recent PDB file in the working directory
            latest_pdb = max(pdb_files, key=lambda f: os.path.getctime(os.path.join(self.working_dir, f)))
            pdb_path = os.path.join(self.working_dir, latest_pdb)
        
        # Extract molecule structure information
        molecules = {}
        current_molecule = None
        current_residue_id = None
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        residue_name = line[17:20].strip()
                        residue_id = line[22:26].strip()
                        atom_name = line[12:16].strip()
                        
                        # Track atoms by molecule type and residue ID
                        if residue_name not in molecules:
                            molecules[residue_name] = []
                        
                        # Only add unique atoms for this molecule type
                        if atom_name not in molecules[residue_name]:
                            molecules[residue_name].append(atom_name)
            
            self.logger.info(f"Analyzed molecule structures: {molecules}")
            return molecules
        except Exception as e:
            self.logger.warning(f"Error analyzing PDB molecule structure: {e}")
            return {}
    def _update_pdb_molecule_names(self, molecule_mappings: Dict[str, str]) -> str:
        """
        Updates molecule names in the PDB file based on mappings from LLM.
        Handles special case of splitting molecules (like NAC into Na and Cl).
        
        :param molecule_mappings: Dictionary mapping current PDB molecule names to desired names.
        :return: Path to updated PDB file.
        """
        self.logger.info(f"Updating PDB molecule names with mappings: {molecule_mappings}")
        
        # Find the latest PDB file in working directory
        pdb_files = [f for f in os.listdir(self.working_dir) if f.endswith('.pdb')]
        if not pdb_files:
            raise FileNotFoundError("No PDB file found in working directory")
        
        # Sort by creation time to get the most recent one
        latest_pdb = max(pdb_files, key=lambda f: os.path.getctime(os.path.join(self.working_dir, f)))
        pdb_path = os.path.join(self.working_dir, latest_pdb)
        
        # Create output path for updated PDB
        updated_pdb_path = os.path.join(self.working_dir, f"mapped_{latest_pdb}")
        
        # Process complex mappings with atom-specific rules
        atom_specific_mappings = {}
        for src, dst in molecule_mappings.items():
            # Check if this is a special mapping with atom-specific rules
            if ":" in dst:
                parts = dst.split(":")
                if len(parts) == 2:
                    atom_type, new_name = parts
                    if src not in atom_specific_mappings:
                        atom_specific_mappings[src] = {}
                    atom_specific_mappings[src][atom_type] = new_name
                    # Remove this entry from regular mappings
                    molecule_mappings.pop(src)
        
        # Read and update the PDB file
        updated_lines = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    residue_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    
                    # Handle standard molecule mappings
                    if residue_name in molecule_mappings:
                        # Replace the residue name with the mapped name (ensuring it's 3 chars max)
                        new_name = molecule_mappings[residue_name][:3]
                        line = line[:17] + f"{new_name:<3}" + line[20:]
                    
                    # Handle atom-specific mappings
                    elif residue_name in atom_specific_mappings:
                        for atom_pattern, new_name in atom_specific_mappings[residue_name].items():
                            # If this atom matches the pattern, apply the specific mapping
                            if atom_pattern == "*" or atom_pattern in atom_name:
                                new_name = new_name[:3]  # Ensure it's 3 chars max
                                line = line[:17] + f"{new_name:<3}" + line[20:]
                                break
                
                updated_lines.append(line)
        
        # Write the updated PDB file
        with open(updated_pdb_path, 'w') as f:
            f.writelines(updated_lines)
        
        self.logger.info(f"Updated PDB file with mapped molecule names: {updated_pdb_path}")
        return updated_pdb_path
