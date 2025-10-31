import os
import re
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import google.generativeai as genai
from .lammps_agent import LAMMPSSimulationAgent

class LAMMPSUpdater:
    """
    Self-evolving updater that analyzes LAMMPS errors and generates solutions.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro-preview-05-06"):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _extract_errors(self, log_content: str) -> List[str]:
        """Extract all errors from LAMMPS output with the following line for context."""
        lines = log_content.splitlines()
        issues = []
        
        for i, line in enumerate(lines):
            if re.search(r"ERROR", line, flags=re.IGNORECASE):
                error_msg = line.strip()
                # Add the next line if it exists
                if i + 1 < len(lines):
                    error_msg += "\nLast command: " + lines[i + 1].strip()
                issues.append(error_msg)
        
        return issues

    def _analyze_issues(self, errors: List[str], input_script: str, data_content: str = "", 
                       ff_content: str = "", log_content: str = "") -> Dict[str, Any]:
        """Have the LLM analyze the issues and suggest a correction strategy."""
        self.logger.info("Analyzing LAMMPS errors and suggesting correction strategy")
        
        analysis_prompt = f"""
        You are a LAMMPS simulation expert. Analyze these LAMMPS errors/warnings, the input script, and log file to:
        1. Identify the root cause of each issue
        2. Suggest specific corrections
        3. Determine what parts of the script need modification
        4. Analyze how far the simulation progressed before the error

        ERRORS/WARNINGS:
        {'\n'.join(errors)}

        INPUT SCRIPT:
        {input_script}

        {'DATA FILE (excerpt):' + data_content if data_content else ''}
        
        {'FORCE FIELD FILE:' + ff_content if ff_content else ''}

        {'LOG FILE EXCERPT:' + log_content[:2000] if log_content else ''}

        Respond with a JSON object with the following structure:
        {{
            "issues": [
                {{
                    "error_text": "The exact error message",
                    "root_cause": "Detailed explanation of what's causing this error",
                    "fix_strategy": "Specific strategy to fix this issue",
                    "script_sections_to_modify": ["section1", "section2"]
                }}
            ],
            "overall_assessment": "Overall assessment of the problems",
            "is_data_file_problem": true/false,
            "is_force_field_problem": true/false,
            "simulation_progress": {{
                "stage": "minimization/equilibration/production",
                "completed_steps": 0,
                "total_steps": 0,
                "percent_complete": 0
            }},
            "correction_approach": "Detailed approach to correct the script",
            "critical_commands_to_add": ["command1", "command2"],
            "critical_commands_to_remove": ["command1", "command2"]
        }}
        """
        
        try:
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(analysis_prompt, generation_config=generation_config)
            analysis = json.loads(response.text)
            self.logger.info(f"Analysis completed: {len(analysis.get('issues', []))} issues identified")
            return analysis
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            # Return a basic analysis structure if parsing fails
            return {
                "issues": [{"error_text": err, "root_cause": "Unknown", 
                           "fix_strategy": "Review LAMMPS documentation", 
                           "script_sections_to_modify": ["Unknown"]} for err in errors],
                "overall_assessment": "Failed to analyze errors properly",
                "is_data_file_problem": False,
                "is_force_field_problem": False,
                "simulation_progress": {
                    "stage": "unknown",
                    "completed_steps": 0,
                    "total_steps": 0,
                    "percent_complete": 0
                },
                "correction_approach": "Manual review needed",
                "critical_commands_to_add": [],
                "critical_commands_to_remove": []
            }

    def _generate_correction_prompt(self, analysis: Dict[str, Any], 
                                  input_script: str, 
                                  research_goal: str,
                                  data_content: str = "", 
                                  ff_content: str = "") -> str:
        """Generate a targeted prompt for script correction based on the analysis."""
        self.logger.info("Generating correction prompt based on analysis")
        
        # Extract key information from analysis
        issues_summary = "\n".join([f"- {issue['error_text']}: {issue['root_cause']}" 
                                  for issue in analysis.get("issues", [])])
        fix_strategies = "\n".join([f"- {issue['fix_strategy']}" 
                                  for issue in analysis.get("issues", [])])
        
        # Extract simulation progress if available
        sim_progress = analysis.get("simulation_progress", {})
        progress_info = ""
        if sim_progress:
            stage = sim_progress.get("stage", "unknown")
            completed = sim_progress.get("completed_steps", 0)
            total = sim_progress.get("total_steps", 0)
            percent = sim_progress.get("percent_complete", 0)
            progress_info = f"\nSimulation progress: {stage} phase, {completed}/{total} steps ({percent}% complete)"
        
        # Build the correction prompt
        correction_prompt = f"""
        As a LAMMPS expert, correct this input script for the research goal: "{research_goal}"
        
        The script has the following issues:
        {issues_summary}
        
        Suggested fix strategies:
        {fix_strategies}
        
        Overall assessment: {analysis.get("overall_assessment", "Unknown")}{progress_info}
        
        Critical commands to add: {', '.join(analysis.get("critical_commands_to_add", []))}
        Critical commands to remove: {', '.join(analysis.get("critical_commands_to_remove", []))}
        
        ORIGINAL INPUT SCRIPT:
        {input_script}
        
        {'DATA FILE EXCERPT:' + data_content if data_content else ''}
        
        {'FORCE FIELD FILE:' + ff_content if ff_content else ''}
        
        Please provide a complete, corrected LAMMPS input script that addresses all the issues. 
        The script should be ready to run without any errors.
        
        IMPORTANT: Make sure to include restart file writing in your script:
        - Write restart files periodically (every 10,000-50,000 steps) during both equilibration and production
        - Use timestep-based naming like "restart.*.equil" for equilibration and "restart.*.prod" for production
       
        IMPORTANT: Only make corrections that address the issues above to ensure a consistent and systematic refinement.
 
        IMPORTANT: Return ONLY the raw LAMMPS script content without any markdown formatting, 
        code block markers, or backticks. Do not include any explanation, just the corrected script.
        """
        
        return correction_prompt

    def _clean_script(self, script_text: str) -> str:
        """Remove markdown formatting and other unwanted elements from the script."""
        # Remove markdown code block markers
        script_text = re.sub(r'```(?:lammps|bash)?', '', script_text)
        # Remove any trailing backticks that might be closing a code block
        script_text = script_text.replace('```', '')
        # Remove any leading/trailing whitespace
        script_text = script_text.strip()
        return script_text

    def refine_inputs(self,
                    input_path: str,
                    research_goal: str,
                    ff_path: Optional[str] = None,
                    data_path: Optional[str] = None,
                    lammps_log: str = 'log.lammps') -> Tuple[str, Dict[str, Any]]:
        """
        Refine LAMMPS input files based on error output using a self-evolving approach.
        
        Args:
            input_path: Path to the original LAMMPS input file
            research_goal: Research goal description
            ff_path: Path to the force field parameter file (optional)
            data_path: Path to the LAMMPS data file (optional)
            lammps_log: Path to the LAMMPS log file with errors
            
        Returns:
            Tuple of (updated script text, analysis results)
        """
        self.logger.info(f"Refining LAMMPS input: {input_path}")
        
        # Read input file
        input_txt = Path(input_path).read_text()
        
        # Read force field file if provided
        ff_txt = ""
        if ff_path and os.path.exists(ff_path):
            ff_txt = Path(ff_path).read_text()
        
        # Read data file if provided (excerpt)
        data_txt = ""
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                # Read first 100 lines to check for structure and charges
                data_txt = "".join(f.readlines()[:100])
        
        # Extract errors and warnings from log
        log_txt = ""
        if os.path.exists(lammps_log):
            with open(lammps_log, 'r') as f:
                log_txt = f.read()
        
        error_list = self._extract_errors(log_txt)
        if not error_list:
            self.logger.warning("No errors or warnings found in log file")
            return input_txt, {"issues": [], "overall_assessment": "No errors found"}
        
        # Check for restart files in the directory
        working_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else "."
        restart_files = list(Path(working_dir).glob('restart.*'))
        
        # Sort restart files by modification time to find the latest one
        has_restart = False
        latest_restart = None
        if restart_files:
            restart_files.sort(key=os.path.getmtime)
            latest_restart = restart_files[-1]
            has_restart = True
            self.logger.info(f"Found latest restart file: {latest_restart}")
        
        # Step 1: Analyze the issues
        analysis = self._analyze_issues(error_list, input_txt, data_txt, ff_txt, log_txt)
        
        # Add restart info to the analysis
        if has_restart:
            analysis["restart_file"] = str(latest_restart)
            analysis["should_restart"] = True
        else:
            analysis["should_restart"] = False
        
        # Step 2: Generate targeted correction prompt
        correction_prompt = self._generate_correction_prompt(
            analysis, input_txt, research_goal, data_txt, ff_txt
        )
        
        # Add restart-specific instructions if needed
        if has_restart:
            restart_basename = os.path.basename(str(latest_restart))
            correction_prompt += f"""
            IMPORTANT: A restart file was found at {latest_restart}. 
            Please modify the script to:
            1. Use this restart file instead of the data file for initial configuration
            2. Comment out the 'read_data' command and uncomment/add the 'read_restart {restart_basename}' command
            3. Skip minimization if it was already completed
            4. Resume the simulation from where it left off

            When modifying the script for restart, follow these CRITICAL rules:
            
            1. COMMAND ORDERING IS CRITICAL for LAMMPS restarts. Follow this exact sequence:
               a. Keep initialization commands (units, dimension, atom_style)
               b. Replace 'read_data' with 'read_restart {restart_basename}'
               c. Define groups AFTER read_restart
               d. Apply 'set' commands (charges, etc.) AFTER read_restart
               e. Define force field parameters (pair_style, bond_style, etc.)
               f. Define 'fix shake' commands BEFORE any box-changing fixes
               g. Define box-changing fixes (fix npt, fix nvt, etc.) AFTER shake
               h. Skip minimization steps that were already completed
               
            2. CRITICAL ERROR PREVENTION:
               - NEVER place 'set' commands before 'read_restart'
               - NEVER place 'fix shake' after box-changing fixes like 'fix npt'
               - Always define groups after the restart file is read
               - Ensure all force field parameters are redefined even with restart
            
            3. RESUME STRATEGY: 
               - If the error occurred in equilibration, skip to production
               - If the error occurred mid-production, continue production from the saved point
            """
        
        # Step 3: Generate corrected script
        self.logger.info("Generating corrected LAMMPS script")
        response = self.model.generate_content(correction_prompt)
        corrected_script = response.text
        
        # Step 4: Clean and format the script
        corrected_script = self._clean_script(corrected_script)
        
        # Step 5: Ensure restart functionality is properly implemented
        if has_restart:
            corrected_script = self._ensure_restart_handling(
                corrected_script, 
                os.path.basename(str(latest_restart))
            )
        
        # Log completion
        self.logger.info(f"Script correction completed - {len(corrected_script)} characters")
        return corrected_script, analysis
    
    def _ensure_restart_handling(self, script_text: str, restart_file: str) -> str:
        """Ensure the script properly uses the restart file."""
        lines = script_text.split('\n')
        has_read_restart = False
        
        for i, line in enumerate(lines):
            # Comment out read_data if found
            if "read_data" in line and not line.strip().startswith('#'):
                lines[i] = f"# {line}  # Commented out for restart"
                
            # Check if read_restart already exists
            if "read_restart" in line and not line.strip().startswith('#'):
                has_read_restart = True
                # If it's there but pointing to a different file, update it
                if restart_file not in line:
                    lines[i] = f"read_restart {restart_file}  # Updated restart file"
        
        # If no read_restart command found, add one
        if not has_read_restart:
            # Find a good place to insert it - after units, atom_style but before region/create_box
            insert_pos = 0
            for i, line in enumerate(lines):
                if any(cmd in line.lower() for cmd in ["units", "atom_style"]):
                    insert_pos = max(insert_pos, i + 1)
                elif "read_data" in line and line.strip().startswith('#'):
                    insert_pos = i
                    break
            
            lines.insert(insert_pos, f"read_restart {restart_file}  # Added for restart")
        
        return '\n'.join(lines)
