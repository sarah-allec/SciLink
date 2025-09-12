import os
import json
import logging
import re
import shutil
import google.generativeai as genai
from ase.build import molecule
from ase.collections import g2
from ase.io import write
from ase.data.pubchem import pubchem_atoms_search
from google.generativeai.types import GenerationConfig
from .instruct import (
    MOLECULE_EXTRACTION_TEMPLATE,
    SMILES_GENERATION_TEMPLATE,
    MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS
)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class MoltemplateInputAgent:
    def __init__(self, api_key=None, model_name="gemini-2.5-pro-preview-05-06", working_dir="moltemplate_run", force_field_dir=None):
        # Model Configuration
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("An API key is required. Pass it as an argument or set the 'GOOGLE_API_KEY' environment variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")

        # Directory Setup
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.force_field_dir = force_field_dir  # Force field directory
        self.force_field_files = self._load_force_field_files(force_field_dir) if force_field_dir else []
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Available force field files: {self.force_field_files}")

    def _load_force_field_files(self, force_field_dir):
        """Load force field files from the specified directory."""
        if not os.path.exists(force_field_dir):
            raise ValueError(f"Force field directory '{force_field_dir}' does not exist.")
        return [
            os.path.join(force_field_dir, f) for f in os.listdir(force_field_dir)
            if f.endswith(".lt")
        ]

    def _validate_force_field_compatibility(self, selected_force_field, molecule_data):
        """
        Ensure compatibility between `system.lt` atom types and the selected force field.

        Args:
            selected_force_field (str): Path to the selected force field file.
            molecule_data (list): Molecule information extracted from the system description.

        Returns:
            dict: Mapping of valid atom types, masses, pair coefficients, etc.
        """
        if not selected_force_field or not os.path.exists(selected_force_field):
            raise ValueError(f"Selected force field file '{selected_force_field}' does not exist.")

        # Load the contents of the selected force field file
        atom_types = {}
        masses = {}
        pair_coeffs = {}
        with open(selected_force_field, 'r') as f:
            force_field_content = f.read()
            # Extract atom types and masses
            mass_pattern = r"@atom:(\w+)\s+([\d\.]+)"
            for match in re.findall(mass_pattern, force_field_content):
                atom_types[match[0]] = f"@atom:{match[0]}"
                masses[match[0]] = float(match[1])

            # Extract pair coefficients (if available)
            pair_coeff_pattern = r"pair_coeff\s+@atom:(\w+)\s+@atom:(\w+)\s+([\d\.]+)\s+([\d\.]+)"
            for match in re.findall(pair_coeff_pattern, force_field_content):
                atom1 = match[0]
                atom2 = match[1]
                epsilon = float(match[2])
                sigma = float(match[3])
                pair_coeffs[(atom1, atom2)] = (epsilon, sigma)

        # Validate molecule atom types against the force field atom types
        for mol in molecule_data:
            identifier = mol.get("identifier")
            atom_type = mol.get("atom_type", None)
            if atom_type not in atom_types:
                self.logger.warning(f"Atom type '{atom_type}' for molecule '{identifier}' not found in force field. "
                                    f"Attempting to auto-map atom type.")
                # Attempt auto-mapping (example logic)
                mapped_atom_type = self._auto_map_atom_type(atom_type, atom_types)
                if mapped_atom_type:
                    mol["atom_type"] = mapped_atom_type
                    self.logger.info(f"Auto-mapped atom type '{atom_type}' to '{mapped_atom_type}'.")
                else:
                    raise ValueError(f"Atom type '{atom_type}' for molecule '{identifier}' could not be validated against "
                                     f"the force field '{selected_force_field}'.")

        return {
            "atom_types": atom_types,
            "masses": masses,
            "pair_coeffs": pair_coeffs
        }

    def _auto_map_atom_type(self, atom_type, atom_types):
        """
        Attempt to auto-map an unsupported atom type to a valid type in the force field.

        Args:
            atom_type (str): Unsupported atom type.
            atom_types (dict): Valid atom types defined by the force field.

        Returns:
            str: A mapped atom type if found, else None.
        """
        # Example: Check for partial matches between atom type and force field atom types
        for valid_atom in atom_types:
            if atom_type.lower() in valid_atom.lower():
                return atom_types[valid_atom]
        return None

    def generate_moltemplate_inputs(self, json_path: str, original_request: str) -> dict:
        """
        Generate the Moltemplate input file (`system.lt`) based on the system description.
    
        Args:
            json_path (str): Path to the JSON file describing the molecular system.
            original_request (str): User-provided description of the simulation goal.
    
        Returns:
            dict: Response with the generated Moltemplate file or an error message.
        """
        # Step 1: Read the JSON file
        try:
            with open(json_path, 'r') as f:
                system_description = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to read JSON file: {e}"}
    
        # Extract "Sample matrix" field
        try:
            sample_matrix = system_description["Sample matrix"]
        except KeyError:
            return {"status": "error", "message": "Missing 'Sample matrix' key in JSON file."}
    
        # Extract molecular components using the LLM
        extraction_prompt = MOLECULE_EXTRACTION_TEMPLATE.format(description=sample_matrix)
        try:
            response = self.model.generate_content(extraction_prompt, generation_config=self.generation_config)
            molecule_data = json.loads(response.text).get("molecules", [])
            if not molecule_data:
                return {"status": "error", "message": "Failed to extract molecular components from 'Sample matrix'."}
        except Exception as e:
            return {"status": "error", "message": f"Molecule extraction failed: {e}"}
    
        # Select the most appropriate force field
        if not self.force_field_files:
            return {"status": "error", "message": "No force field files available for selection."}
    
        selection_prompt = f"""
        Based on the following molecular system description:
        "{sample_matrix}"
        Select the most appropriate force field file from the following options:
        {', '.join(self.force_field_files)}
        Provide the name of the selected file and explain the choice.
        """
        try:
            selection_response = self.model.generate_content(selection_prompt, generation_config=self.generation_config)
            # Parse the response as JSON and extract the file path
            selection_data = json.loads(selection_response.text)
            force_field_selection = selection_data.get("selected_file")
            if not force_field_selection or not os.path.exists(force_field_selection):
                raise ValueError(f"Selected force field file '{selection_data}' does not exist.")
            self.logger.info(f"Selected force field: {force_field_selection}")
        except Exception as e:
            return {"status": "error", "message": f"Force field selection failed: {e}"}
    
        # Validate compatibility between system.lt and the selected force field
        try:
            force_field_compatibility = self._validate_force_field_compatibility(force_field_selection, molecule_data)
        except Exception as e:
            return {"status": "error", "message": f"Force field compatibility validation failed: {e}"}
    
        # Generate Moltemplate input file
        try:
            moltemplate_prompt = MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS.format(
                system_description=sample_matrix,
                molecule_list=", ".join([f"{mol['identifier']}" for mol in molecule_data]),
                original_request=original_request,
                force_field=force_field_selection,
                force_field_files=", ".join(self.force_field_files)
            )
            response = self.model.generate_content(moltemplate_prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            result["status"] = "success"
            result["selected_force_field"] = force_field_selection
            return result
        except Exception as e:
            return {"status": "error", "message": f"Moltemplate input generation failed: {e}"}
