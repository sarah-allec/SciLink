import os
import json
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from .instruct import LAMMPS_INPUT_GENERATION_INSTRUCTIONS


class LammpsInputAgent:
    """Agent for generating LAMMPS and pdb2dat input files."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-pro-preview-05-06"):
        # Check for the API key argument or fallback to environment variable
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")  # Get from env if not provided
        if not api_key:  # Raise error if still no API key
            raise ValueError("API key required. Pass it as an argument or set the 'GOOGLE_API_KEY' environment variable.")
        
        # Configure the model with the API key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.logger = logging.getLogger(__name__)

    def generate_lammps_inputs(self, json_path: str, pdb_file_paths: list, system_pdb: str,original_request: str) -> dict:
        """Generate LAMMPS and pdb2dat input files."""
        
        # Read JSON file
        try:
            with open(json_path, 'r') as f:
                system_description = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to read JSON file: {e}"}

        # Extract the "Sample matrix" from the system description JSON
        try:
            sample_matrix = system_description["Sample matrix"]
        except KeyError as e:
            return {"status": "error", "message": f"Missing key in JSON file: {e}"}

        # Read PDB files
        pdb_contents = {}
        try:
            for pdb_file in pdb_file_paths:
                with open(pdb_file, 'r') as f:
                    pdb_contents[pdb_file] = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read XYZ file: {e}"}

        # Build prompt
        formatted_pdb_files = "\n".join([f"--- {file_name} ---\n{file_content}"
                                         for file_name, file_content in pdb_contents.items()])
        prompt = LAMMPS_INPUT_GENERATION_INSTRUCTIONS.format(
            sample_matrix=sample_matrix,
            pdb_files=formatted_pdb_files,
            original_request=original_request,
            packmol_output=system_pdb
        )

        # Get LLM response
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": f"Generation failed: {e}"}

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """Save input files."""
        if result.get("status") != "success":
            return {"error": "Generation was not successful"}
        
        os.makedirs(output_dir, exist_ok=True)
        saved = {}

        try:
            # Save in.lmp
            with open(os.path.join(output_dir, "in.lmp"), 'w') as f:
                f.write(result["input"])
            saved["input"] = os.path.join(output_dir, "in.lmp")
            
            # Save settings.py
            with open(os.path.join(output_dir, "settings.py"), 'w') as f:
                f.write(result["pdb2dat"])
            saved["pdb2dat"] = os.path.join(output_dir, "settings.py")
            
            return saved
        except Exception as e:
            return {"error": f"Save failed: {e}"}

    def apply_improvements(self, original_input: str, validation_result: dict,
                           json_path: str, pdb_file_paths: list, original_request: str,
                           output_dir: str = ".") -> dict:
        """Regenerate input file using LLM with improvement instructions."""
        
        if validation_result.get("validation_status") != "needs_adjustment":
            return {
                "status": "no_changes",
                "message": "No improvements needed - in.lmp is already good"
            }

        adjustments = validation_result.get("suggested_adjustments", [])
        if not adjustments:
            return {"status": "error", "message": "No adjustments available"}

        # Read system description (JSON)
        try:
            with open(json_path, 'r') as f:
                system_description = json.load(f)
            sample_matrix = system_description.get("Sample matrix", "[Sample matrix not specified]")
        except Exception as e:
            return {"status": "error", "message": f"Failed to read JSON file: {e}"}

        # Read PDB files
        pdb_contents = {}
        try:
            for pdb_file in pdb_file_paths:
                with open(pdb_file, 'r') as f:
                    pdb_contents[pdb_file] = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read XYZ file: {e}"}

        # Formatting improvement instructions
        improvement_instructions = "IMPROVEMENT INSTRUCTIONS:\n"
        improvement_instructions += "Please modify the provided in.lmp file based on these literature-validated suggestions:\n\n"
        for adj in adjustments:
            improvement_instructions += f"• {adj.get('parameter')}: {adj.get('current_value')} → {adj.get('suggested_value')}\n"
            improvement_instructions += f"  Reason: {adj.get('reason')}\n\n"
        improvement_instructions += f"Literature assessment: {validation_result.get('overall_assessment', '')}\n\n"
        improvement_instructions += "Generate an improved in.lmp file incorporating these changes."

        # Build the prompt with original input, improvement instructions, and XYZ content
        formatted_pdb_files = "\n".join([f"--- {file_name} ---\n{file_content}"
                                         for file_name, file_content in pdb_contents.items()])
        prompt = f"""{LAMMPS_INPUT_GENERATION_INSTRUCTIONS}
## ORIGINAL IN.LMP TO IMPROVE:
{original_input}
## {improvement_instructions}
## SYSTEM DESCRIPTION:
{sample_matrix}
## PDB FILE CONTENTS:
{formatted_pdb_files}
## ORIGINAL SYSTEM DESCRIPTION:
{original_request}
Please generate an improved in.lmp file based on the improvement instructions above."""

        # Get improved in.lmp from LLM
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            if result.get("input"):
                # Save improved in.lmp
                os.makedirs(output_dir, exist_ok=True)
                improved_path = os.path.join(output_dir, "in_improved.lmp")
                with open(improved_path, 'w') as f:
                    f.write(result["input"])
                result.update({
                    "status": "success",
                    "improvements_applied": True,
                    "adjustments_count": len(adjustments),
                    "improved_input_path": improved_path
                })
                self.logger.info(f"Generated improved in.lmp with {len(adjustments)} literature-based improvements")
                return result
            else:
                return {"status": "error", "message": "No in.lmp generated in LLM response"}
        except Exception as e:
            self.logger.error(f"Failed to generate improved in.lmp: {e}")
            return {"status": "error", "message": f"Failed to generate improved in.lmp: {e}"}
