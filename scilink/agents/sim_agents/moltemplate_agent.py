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
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("An API key is required. Pass it as an argument or set the 'GOOGLE_API_KEY' environment variable.")

        # Configure the model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")

        # Initialize the working directory
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        self.force_field_dir = force_field_dir  # Optional force field directory path
        self.force_field_files = self._load_force_field_files(force_field_dir) if force_field_dir else []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Available force field files: {self.force_field_files}")

    def _load_force_field_files(self, force_field_dir):
        """Load available `.lt` files in the specified force field directory."""
        if not os.path.exists(force_field_dir):
            raise ValueError(f"Force field directory '{force_field_dir}' does not exist.")
        return [file for file in os.listdir(force_field_dir) if file.endswith(".lt")]

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

        # Step 2: Extract "Sample matrix" field
        try:
            sample_matrix = system_description["Sample matrix"]
        except KeyError:
            return {"status": "error", "message": "Missing 'Sample matrix' key in JSON file."}

        # Step 3: Extract molecular components using the LLM
        extraction_prompt = MOLECULE_EXTRACTION_TEMPLATE.format(description=sample_matrix)
        try:
            response = self.model.generate_content(extraction_prompt, generation_config=self.generation_config)
            molecule_data = json.loads(response.text).get("molecules", [])
            if not molecule_data:
                return {"status": "error", "message": "Failed to extract molecular components from 'Sample matrix'."}
        except Exception as e:
            return {"status": "error", "message": f"Molecule extraction failed: {e}"}

        # Step 4: Select the most appropriate force field
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
            force_field_selection = selection_response.text.strip()
            if not force_field_selection:
                return {"status": "error", "message": "Failed to select an appropriate force field."}
            self.logger.info(f"Selected force field: {force_field_selection}")
        except Exception as e:
            return {"status": "error", "message": f"Force field selection failed: {e}"}

        # Step 5: Generate Moltemplate input file
        try:
            moltemplate_prompt = MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS.format(
                system_description=sample_matrix,
                molecule_list=", ".join([f"{mol['identifier']}" for mol in molecule_data]),
                original_request=original_request,
                force_field=force_field_selection,
                force_field_files=", ".join(self.force_field_files)  # Include available force field files
            )
            response = self.model.generate_content(moltemplate_prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            result["status"] = "success"
            result["selected_force_field"] = force_field_selection
            return result
        except Exception as e:
            return {"status": "error", "message": f"Moltemplate input generation failed: {e}"}

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """
        Save the generated Moltemplate input files (`system.lt`) to the specified directory.
        Args:
            result (dict): Response from `generate_moltemplate_inputs`.
            output_dir (str): Directory where files will be saved. Default is current directory.
        Returns:
            dict: A dictionary with saved file paths or an error message.
        """
        if result.get("status") != "success":
            return {"status": "error", "message": "Moltemplate input generation was not successful."}

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        errors = []

        try:
            # Save the `system.lt` file
            if "system_lt" in result:
                lt_filepath = os.path.join(output_dir, "system.lt")
                with open(lt_filepath, 'w') as f:
                    f.write(result["system_lt"])
                saved_files["system_lt"] = lt_filepath

            # Save additional outputs if present
            for key, content in result.items():
                if key in ["system_lt", "status"] or not isinstance(content, str):
                    continue
                try:
                    filepath = os.path.join(output_dir, f"{key}.txt")
                    with open(filepath, 'w') as f:
                        f.write(content)
                    saved_files[key] = filepath
                except Exception as e:
                    errors.append({key: str(e)})
                    continue

            return {"status": "success", "saved_files": saved_files, "errors": errors if errors else None}
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": f"Failed to save files due to a critical error: {e}",
                "traceback": traceback.format_exc()
            }
