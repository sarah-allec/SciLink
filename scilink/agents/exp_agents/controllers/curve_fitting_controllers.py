import logging
import json
import re
import numpy as np
import os
from typing import Callable

from ..preprocess import CurvePreprocessingAgent
from ....executors import ScriptExecutor
from ...lit_agents.literature_agent import FittingModelLiteratureAgent
from ....tools.curve_fitting_tools import plot_curve_to_bytes
from ..instruct import (
    LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
    FITTING_SCRIPT_GENERATION_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
    FITTING_QUALITY_ASSESSMENT_INSTRUCTIONS,
    FITTING_MODEL_CORRECTION_INSTRUCTIONS
)

# --- Tool Controllers ---

class RunCurvePreprocessingController:
    """
    [🛠️ Tool Step]
    Runs the CurvePreprocessingAgent on the initial data.
    """
    def __init__(self, logger: logging.Logger, preprocessor: CurvePreprocessingAgent, output_dir: str):
        self.logger = logger
        self.preprocessor = preprocessor
        self.output_dir = output_dir

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: Curve Preprocessing --- 🛠️\n")
        
        try:
            # The preprocessor is already refactored and handles its own logic
            processed_data, data_quality = self.preprocessor.run_preprocessing(
                state["curve_data"], state["system_info"]
            )
            
            # Overwrite with processed data
            state["curve_data"] = processed_data
            state["data_quality"] = data_quality
            
            # Save processed data to a temp file for the script executor
            # Use PID to avoid conflicts in concurrent runs
            pid = os.getpid()
            processed_data_path = os.path.join(self.output_dir, f"temp_processed_curve_data_{pid}.npy")
            np.save(processed_data_path, processed_data)
            state["processed_data_path"] = processed_data_path
            
            self.logger.info(f"✅ Tool Complete: Curve preprocessing finished. Temp data at {processed_data_path}")
        
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Curve preprocessing: {e}", exc_info=True)
            state["error_dict"] = {"error": "Curve preprocessing failed", "details": str(e)}
        return state

class CreateInitialPlotController:
    """
    [🛠️ Tool Step]
    Plots the (potentially processed) curve to provide context for later steps.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: Create Initial Plot --- 🛠️\n")
        try:
            plot_bytes = plot_curve_to_bytes(
                state["curve_data"], state["system_info"], " (Processed Data)"
            )
            state["original_plot_bytes"] = plot_bytes
            # This is the first image, so it replaces any previous
            state["analysis_images"] = [
                {'label': 'Processed Data', 'data': plot_bytes}
            ]
            self.logger.info("✅ Tool Complete: Initial data plot created.")
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Plotting initial curve: {e}", exc_info=True)
            state["error_dict"] = {"error": "Failed to plot initial curve", "details": str(e)}
        return state

class RunLiteratureSearchController:
    """
    [🛠️ Tool Step]
    Runs the literature search using the query from the state.
    """
    def __init__(self, logger: logging.Logger, literature_agent: FittingModelLiteratureAgent | None, output_dir: str):
        self.logger = logger
        self.literature_agent = literature_agent
        self.output_dir = output_dir

    def _save_literature_step_results(self, query: str, report: str) -> dict:
        """Saves the literature search query and the resulting report to files."""
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature_step")
            os.makedirs(lit_dir, exist_ok=True)

            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, 'w') as f:
                f.write(query)
            saved_files["query_file"] = query_path

            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            saved_files["report_file"] = report_path
            
            self.logger.info(f"Saved literature results to: {lit_dir}")
        except Exception as e:
            self.logger.error(f"Failed to save literature step results: {e}")
        return saved_files

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: Literature Search --- 🛠️\n")
        
        lit_query = state.get("literature_query", "N/A (Query generation failed)")
        
        if self.literature_agent is None:
            self.logger.warning("Literature agent not available. Using LLM's internal knowledge.")
            state["literature_context"] = "Literature agent not available. Using LLM's internal knowledge for model selection."
            state["result_json"] = {"literature_files": self._save_literature_step_results(lit_query, state["literature_context"])}
            return state

        try:
            lit_result = self.literature_agent.query_for_models(lit_query)

            if lit_result["status"] == "success":
                literature_context = lit_result["formatted_answer"]
                self.logger.info("✅ Literature search successful.")
            else:
                warning_message = f"Literature search failed ({lit_result['message']}). Falling back to LLM's internal knowledge."
                self.logger.warning(warning_message)
                literature_context = "The external literature search failed. Fall back to your internal knowledge to propose a suitable physical fitting model."
            
            saved_files = self._save_literature_step_results(lit_query, literature_context)
            state["literature_context"] = literature_context
            state["result_json"] = {"literature_files": saved_files} # Per user request
        
        except Exception as lit_e:
            self.logger.error(f"Error during literature search step: {lit_e}", exc_info=True)
            literature_context = "An error occurred during the literature search. Fall back to your internal knowledge to propose a suitable physical fitting model."
            saved_files = self._save_literature_step_results(lit_query, f"Search Error: {lit_e}")
            state["literature_context"] = literature_context
            state["result_json"] = {"literature_files": saved_files}

        return state

class RunFittingLoopController:
    """
    [🛠️/🧠 Meta-Controller]
    Runs the entire multi-attempt fitting loop, including script generation,
    execution, error correction, and fit validation.
    """
    MAX_SCRIPT_ATTEMPTS = 3
    MAX_MODEL_ATTEMPTS = 3

    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable, executor: ScriptExecutor):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.executor = executor

    def _extract_script(self, script_content: str) -> str:
        """Extracts Python code from LLM response."""
        script_content = script_content.strip()
        match = re.search(r"```(?:python)?\n(.*?)\n```", script_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if script_content.lower().startswith("python"):
            potential_code = script_content[len("python"):].strip()
            if potential_code.startswith(("import ", "def ", "#")):
                return potential_code
        if script_content.startswith(("import ", "def ", "#")):
            return script_content
        self.logger.error(f"LLM response did not contain a recognizable Python code block: {script_content[:300]}")
        raise ValueError("LLM failed to generate Python script in a recognizable format.")

    def _generate_fitting_script(self, state: dict, context: str) -> str:
        self.logger.info("Generating initial fitting script...")
        data_preview = np.array2string(state["curve_data"][:10], precision=4, separator=', ')
        prompt = (
            f"{FITTING_SCRIPT_GENERATION_INSTRUCTIONS}\n"
            f"## Literature Context\n{context}\n"
            f"## Curve Data Preview\n{data_preview}\n"
            f"## Data File Path\nThe script should load data from this absolute path: '{os.path.abspath(state['processed_data_path'])}'"
        )
        response = self.model.generate_content(prompt)
        return self._extract_script(response.text)

    def _correct_fitting_script(self, state: dict, context: str, old_script: str, error: str) -> str:
        self.logger.warning("Requesting script correction from LLM...")
        correction_prompt = FITTING_SCRIPT_CORRECTION_INSTRUCTIONS.format(
            literature_context=context,
            failed_script=old_script,
            error_message=error
        )
        response = self.model.generate_content(correction_prompt)
        return self._extract_script(response.text)

    def _evaluate_fit_quality(self, state: dict, plot_bytes: bytes, context: str) -> dict:
        self.logger.info("🤖 Assessing the quality of the curve fit...")
        prompt = [
            FITTING_QUALITY_ASSESSMENT_INSTRUCTIONS,
            "## Original Data Plot", {"mime_type": "image/jpeg", "data": state["original_plot_bytes"]},
            "## Fit Visualization", {"mime_type": "image/png", "data": plot_bytes},
            "## Literature Context\n" + context
        ]
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result_json, error = self._parse_llm_response(response)
        
        if error or not result_json:
            self.logger.warning("Failed to get a valid fit quality assessment from LLM. Assuming fit is acceptable.")
            return {"is_good_fit": True, "critique": "Assessment failed.", "suggestion": "N/A"}
        return result_json

    def _correct_fitting_model(self, state: dict, old_script: str, plot_bytes: bytes, critique: str, suggestion: str, context: str) -> str:
        self.logger.warning("⚠️ Fit was inadequate. Requesting a new model and script from LLM...")
        prompt = [
            FITTING_MODEL_CORRECTION_INSTRUCTIONS,
            "## Critique of Previous Attempt\n" + critique,
            "## Suggestion for Improvement\n" + suggestion,
            "## Plot of the Bad Fit", {"mime_type": "image/png", "data": plot_bytes},
            "## Original Literature Context\n" + context,
            "## Old Script That Produced the Bad Fit\n```python\n" + old_script + "\n```"
        ]
        response = self.model.generate_content(prompt)
        return self._extract_script(response.text)

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️/🧠 --- META-CONTROLLER: Running Fitting Loop --- 🛠️/🧠\n")
        
        literature_context = state["literature_context"]
        fit_data_path = state["processed_data_path"] # Path to the .npy file
        output_dir = os.path.dirname(fit_data_path)
        
        fitting_script = None
        exec_result = None
        fit_plot_bytes = None

        for model_attempt in range(1, self.MAX_MODEL_ATTEMPTS + 1):
            self.logger.info(f"--- Fitting Model Attempt {model_attempt}/{self.MAX_MODEL_ATTEMPTS} ---")
            
            # --- Inner Script-Correction Loop ---
            last_script_error = "No script generated yet."
            script_success = False
            for script_attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
                self.logger.info(f"--- Script Execution Attempt {script_attempt}/{self.MAX_SCRIPT_ATTEMPTS} ---")
                try:
                    if script_attempt == 1:
                        if model_attempt == 1:
                            fitting_script = self._generate_fitting_script(state, literature_context)
                        # else: fitting_script was already set by _correct_fitting_model
                    else:
                        fitting_script = self._correct_fitting_script(state, literature_context, fitting_script, last_script_error)

                    # Execute script
                    exec_result = self.executor.execute_script(fitting_script, working_dir=output_dir)
                    
                    if exec_result.get("status") == "success":
                        self.logger.info("✅ Script executed successfully.")
                        script_success = True
                        break # Exit inner script-correction loop
                    else:
                        last_script_error = exec_result.get("message", "Unknown error")
                        self.logger.warning(f"Script failed: {last_script_error}")
                
                except Exception as e:
                    last_script_error = str(e)
                    self.logger.error(f"Script generation/execution failed: {e}", exc_info=True)

            if not script_success:
                state["error_dict"] = {"error": f"Failed to generate a working script after {self.MAX_SCRIPT_ATTEMPTS} attempts.", "details": last_script_error}
                return state
            
            # --- Fit Quality Assessment ---
            fit_plot_path = os.path.join(output_dir, "fit_visualization.png")
            if not os.path.exists(fit_plot_path):
                state["error_dict"] = {"error": "Script succeeded but did not create 'fit_visualization.png'."}
                return state
                
            with open(fit_plot_path, "rb") as f:
                fit_plot_bytes = f.read()
            
            assessment = self._evaluate_fit_quality(state, fit_plot_bytes, literature_context)
            self.logger.info(f"Fit Assessment Critique: {assessment['critique']}")

            if assessment.get("is_good_fit", False):
                self.logger.info("✅ Fit quality is acceptable. Exiting loop.")
                break # Exit outer model-correction loop
            
            # --- Model Correction ---
            if model_attempt < self.MAX_MODEL_ATTEMPTS:
                literature_context += f"\n\n--- CRITIQUE OF ATTEMPT {model_attempt} ---\nCritique: {assessment['critique']}\nSuggestion: {assessment.get('suggestion', 'N/A')}"
                try:
                    fitting_script = self._correct_fitting_model(
                        state, fitting_script, fit_plot_bytes, 
                        assessment['critique'], assessment.get('suggestion', 'N/A'), literature_context
                    )
                except Exception as e:
                    self.logger.error(f"Model correction script generation failed: {e}")
                    state["error_dict"] = {"error": "Failed to generate correction script", "details": str(e)}
                    return state
            else:
                self.logger.warning("⚠️ Max model attempts reached. Using last fit despite imperfections.")
                break # Exit outer loop

        # --- Store Final Results in State ---
        state["final_fitting_script"] = fitting_script
        state["final_fit_plot_bytes"] = fit_plot_bytes
        state["analysis_images"].append(
            {'label': 'Final Fit Visualization', 'data': fit_plot_bytes}
        )
        
        # Parse parameters from script stdout
        fit_params = {}
        for line in exec_result.get("stdout", "").splitlines():
            if line.startswith("FIT_RESULTS_JSON:"):
                try:
                    fit_params = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                    break
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse FIT_RESULTS_JSON: {e}")
        
        if not fit_params:
            self.logger.warning("Could not parse fitting parameters from script output.")

        # Per user request, add to result_json
        if "result_json" not in state:
            state["result_json"] = {}
        state["result_json"]["fitting_parameters"] = fit_params
            
        return state

# --- LLM Controllers ---

class GetLiteratureQueryController:
    """
    [🧠 LLM Step]
    Uses an LLM to formulate a query for the literature agent.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: Generate Literature Query --- 🧠\n")
        
        try:
            prompt = [
                LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
                "## Data Plot", {"mime_type": "image/jpeg", "data": state["original_plot_bytes"]},
                "\n\nAdditional System Information (Metadata):\n" + json.dumps(state["system_info"], indent=2)
            ]
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result_json, error = self._parse_llm_response(response)
            
            if error or "search_query" not in result_json:
                self.logger.error(f"Failed to generate literature query: {error or 'No search_query key'}")
                state["literature_query"] = "N/A (Query generation failed)"
            else:
                state["literature_query"] = result_json["search_query"]
                self.logger.info(f"✅ LLM Step Complete: Generated query: {state['literature_query']}")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Generate literature query: {e}", exc_info=True)
            state["literature_query"] = "N/A (Exception)"
            
        return state

# --- Prep Controllers ---

class BuildCurveFittingPromptController:
    """
    [📝 Prep Step]
    Gathers all analysis results into the final prompt for interpretation.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n📝 --- PREP STEP: Building Final Interpretation Prompt --- 📝\n")
        
        try:
            prompt_parts = [
                state["instruction_prompt"],
                "\n## Original Data Plot", {"mime_type": "image/jpeg", "data": state["original_plot_bytes"]},
                "\n## Final Fit Visualization", {"mime_type": "image/png", "data": state["final_fit_plot_bytes"]},
                "\n## Final Fitted Parameters\n" + json.dumps(state["result_json"].get("fitting_parameters", {}), indent=2),
                "\n## Final Literature Context\n" + state["literature_context"],
                "\n\nAdditional System Information (Metadata):\n" + json.dumps(state["system_info"], indent=2),
                "\n\nProvide your interpretation in the requested JSON format."
            ]
            
            state["final_prompt_parts"] = prompt_parts
            self.logger.info("✅ Prep Step Complete: Final prompt is ready.")
        
        except Exception as e:
            self.logger.error(f"❌ Prep Step Failed: Prompt building failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Failed to build final prompt", "details": str(e)}
            
        return state