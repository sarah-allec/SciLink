import os
import logging
import json
import re
import shutil
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from .instruct import MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS

class MoltemplateAgent:
    """
    Agent that processes PDB files to generate LAMMPS input via Moltemplate.
    Handles PDB reformatting, atom reordering, system.lt generation, and Moltemplate execution.
    """
    def __init__(self, working_dir: str, ff_library_path: str, api_key: str = None, 
                model_name: str = "gemini-2.5-pro-preview-05-06"):
        self.working_dir = working_dir
        self.ff_library_path = ff_library_path
        os.makedirs(working_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Provide explicitly or set GOOGLE_API_KEY.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")

    def _list_force_fields(self) -> List[str]:
        return sorted([f for f in os.listdir(self.ff_library_path) if f.endswith(".lt")])
    
    def generate_lammps_input(self, description: str, system_pdb: str) -> dict:
        """
        Orchestrates the full workflow from PDB file to LAMMPS input.
        """
        try:
            if not os.path.exists(system_pdb):
                raise FileNotFoundError(f"Input PDB file not found: '{system_pdb}'")
            
            # Step 1: Update and reorder PDB
            updated_pdb_path = self._update_and_reorder_pdb(system_pdb)
            
            # Step 2: Count molecules in the updated PDB
            molecule_counts = self._count_molecules(updated_pdb_path)
            
            # Step 3: Select force field
            selected_ff_name = "spce_oplsaa2024.lt"  # Simplified selection
            selected_ff_path = os.path.join(self.ff_library_path, selected_ff_name)
            if not os.path.exists(selected_ff_path):
                raise FileNotFoundError(f"Force field not found: {selected_ff_path}")
            
            # Step 4: Generate system.lt
            system_lt_path = self._generate_system_lt_with_llm(description, molecule_counts, selected_ff_path)
            
            # Step 5: Run Moltemplate
            lammps_files = self._run_moltemplate(updated_pdb_path, system_lt_path)
            
            return {
                "status": "success",
                "working_dir": self.working_dir,
                "updated_pdb": updated_pdb_path,
                "system_lt": system_lt_path,
                "lammps_data": lammps_files.get("data_file"),
                "message": "LAMMPS input files generated successfully."
            }

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _update_and_reorder_pdb(self, input_pdb: str) -> str:
        """
        Updates residue/atom names, reorders atoms, and saves to the working directory.
        """
        output_pdb = os.path.join(self.working_dir, "system.pdb")
        
        with open(input_pdb, 'r') as f:
            lines = f.readlines()
            
        # Separate lines
        header_lines = [line for line in lines if not line.startswith(("ATOM", "HETATM", "END"))]
        atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
        footer_lines = [line for line in lines if line.startswith("END")]
        
        na_atoms, cl_atoms, water_molecules = [], [], {}
        
        # Categorize atoms
        for line in atom_lines:
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_id = line[22:26].strip()
            
            if residue_name == "MOL" and chain_id == "A":
                if atom_name == "Na":
                    na_atoms.append(line)
                elif atom_name == "Cl":
                    cl_atoms.append(line)
            elif residue_name == "MOL" and chain_id == "B":
                mol_id = f"{chain_id}_{residue_id}"
                if mol_id not in water_molecules:
                    water_molecules[mol_id] = []
                water_molecules[mol_id].append(line)

        # Process and reorder
        processed_lines = []
        
        # Sodium ions
        for line in na_atoms:
            new_line = line[:12] + f"{'Na':<4}" + line[16:17] + f"{'NaI':<3}" + line[20:]
            processed_lines.append(new_line)
        
        # Chloride ions
        for line in cl_atoms:
            new_line = line[:12] + f"{'Cl':<4}" + line[16:17] + f"{'ClI':<3}" + line[20:]
            processed_lines.append(new_line)
            
        # Water molecules
        for mol_id in sorted(water_molecules.keys()):
            atoms = water_molecules[mol_id]
            h_counter = 0
            for line in atoms:
                atom_name = line[12:16].strip()
                if atom_name == 'O':
                    new_atom_name = 'o'
                elif atom_name == 'H':
                    h_counter += 1
                    new_atom_name = f'h{h_counter}'
                else:
                    new_atom_name = atom_name
                
                new_line = line[:12] + f"{new_atom_name:<4}" + line[16:17] + f"{'SPC':<3}" + line[20:]
                processed_lines.append(new_line)

        # Renumber atoms
        renumbered_lines = []
        for i, line in enumerate(processed_lines, 1):
            renumbered_lines.append(line[:6] + f"{i:5d}" + line[11:])
        
        # Save updated PDB
        with open(output_pdb, 'w') as f:
            f.writelines(header_lines)
            f.writelines(renumbered_lines)
            f.writelines(footer_lines)
            
        return output_pdb

    def _count_molecules(self, pdb_path: str) -> Dict[str, int]:
        unique_residues = {}
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_id = line[22:26].strip()
                    unique_id = f"{chain_id}_{residue_id}"
                    
                    if residue_name not in unique_residues:
                        unique_residues[residue_name] = set()
                    unique_residues[residue_name].add(unique_id)
                    
        return {name: len(ids) for name, ids in unique_residues.items()}

    def _generate_system_lt_with_llm(self, description: str, counts: Dict[str, int], 
                                     ff_path: str) -> str:
        self.logger.info("Generating system.lt file with LLM...")
        
        prompt = MOLTEMPLATE_INPUT_GENERATION_INSTRUCTIONS.format(
            system_description=description,
            force_field_path=ff_path,
            molecule_counts=json.dumps(counts)
        )
        
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result = json.loads(response.text)
        
        system_lt_path = os.path.join(self.working_dir, "system.lt")
        with open(system_lt_path, 'w') as f:
            f.write(result["system_lt"])
            
        return system_lt_path

    def _run_moltemplate(self, pdb_file: str, system_lt_file: str) -> Dict[str, Any]:
        self.logger.info("Running Moltemplate...")
        
        orig_dir = os.getcwd()
        os.chdir(self.working_dir)
        
        try:
            cmd = f"moltemplate.sh -pdb {os.path.basename(pdb_file)} {os.path.basename(system_lt_file)}"
            
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True, timeout=300
            )
            
            data_file = os.path.join(os.getcwd(), "system.data")
            if not os.path.exists(data_file):
                raise FileNotFoundError("Moltemplate ran, but 'system.data' was not created.")
                
            return {"data_file": os.path.abspath(data_file)}
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Moltemplate failed. Stderr:\n{e.stderr}")
            raise RuntimeError(f"Moltemplate execution failed: {e.stderr}")
        finally:
            os.chdir(orig_dir)
