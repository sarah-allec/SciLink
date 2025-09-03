import os
import json
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .instruct import LAMMPS_INPUT_GENERATION_INSTRUCTIONS


class LammpsInputAgent:
    """Agent for generating LAMMPS data and input files."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro-preview-05-06"):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.logger = logging.getLogger(__name__)

    def generate_lammps_inputs(self, pdb_path: str, original_request: str) -> dict:
        """Generate LAMMPS data and input files."""
        
        # Read PDB file
        try:
            with open(pdb_path, 'r') as f:
                pdb_content = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read PDB file: {e}"}

        # Build prompt
        prompt = LAMMPS_INPUT_GENERATION_INSTRUCTIONS.format(
            pdb_content=pdb_content,
            original_request=original_request
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
        """Save input and data files."""
        if result.get("status") != "success":
            return {"error": "Generation was not successful"}
        
        os.makedirs(output_dir, exist_ok=True)
        saved = {}
        
        try:
            # Save in.lmp
            with open(os.path.join(output_dir, "in.lmp"), 'w') as f:
                f.write(result["input"])
            saved["input"] = os.path.join(output_dir, "in.lmp")
            
            # Save lmp.data 
            with open(os.path.join(output_dir, "lmp.data"), 'w') as f:
                f.write(result["data"])
            saved["data"] = os.path.join(output_dir, "lmp.data")

            return saved
        
        except Exception as e:
            return {"error": f"Save failed: {e}"}
             
    def apply_improvements(self, original_input: str, validation_result: dict, 
                          pdb_path: str, original_request: str, output_dir: str = ".") -> dict:
        """Regenerate input file using LLM with improvement instructions."""
        
        if validation_result.get("validation_status") != "needs_adjustment":
            return {
                "status": "no_changes", 
                "message": "No improvements needed - in.lmp is already good"
            }
        
        adjustments = validation_result.get("suggested_adjustments", [])
        if not adjustments:
            return {"status": "error", "message": "No adjustments available"}
        
        # Read PDB content
        try:
            with open(pdb_path, 'r') as f:
                pdb_content = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read PDB file: {e}"}
        
        # Build improvement instructions
        improvement_instructions = "IMPROVEMENT INSTRUCTIONS:\n"
        improvement_instructions += "Please modify the provided in.lmp file based on these literature-validated suggestions:\n\n"
        
        for adj in adjustments:
            improvement_instructions += f"• {adj.get('parameter')}: {adj.get('current_value')} → {adj.get('suggested_value')}\n"
            improvement_instructions += f"  Reason: {adj.get('reason')}\n\n"
        
        improvement_instructions += f"Literature assessment: {validation_result.get('overall_assessment', '')}\n\n"
        improvement_instructions += "Generate an improved in.lmp file incorporating these changes."
        
        # Build the prompt with original input and improvement instructions
        prompt = f"""{LAMMPS_INPUT_GENERATION_INSTRUCTIONS}

## ORIGINAL IN.LMP TO IMPROVE:
{original_input}

## {improvement_instructions}

## PDB STRUCTURE:
{pdb_content}

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
