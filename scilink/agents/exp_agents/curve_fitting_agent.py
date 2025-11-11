import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json
import os
import re
import logging

from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from ...executors import ScriptExecutor
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .instruct import (
    LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
    FITTING_SCRIPT_GENERATION_INSTRUCTIONS,
    FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
    FITTING_QUALITY_ASSESSMENT_INSTRUCTIONS,
    FITTING_MODEL_CORRECTION_INSTRUCTIONS,
    CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .preprocess import CurvePreprocessingAgent


logger = logging.getLogger(__name__)

class CurveFittingAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing 1D curves via automated, literature-informed fitting.
    """

    MAX_SCRIPT_ATTEMPTS = 3
    MAX_MODEL_ATTEMPTS = 3

    def __init__(self, google_api_key: str = None, futurehouse_api_key: str = None, 
                 model_name: str = "gemini-2.5-pro-preview-06-05", local_model: str = None, 
                 run_preprocessing: bool = True,
                 enable_human_feedback: bool = True, executor_timeout: int = 60, 
                 output_dir: str = "curve_analysis_output", max_wait_time: int = 600, **kwargs):
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        self.executor = ScriptExecutor(timeout=executor_timeout, enforce_sandbox=False)
        self.output_dir = output_dir
        self.literature_agent = None
        effective_api_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")

        self.run_preprocessing = run_preprocessing
        if self.run_preprocessing: 
            self.preprocessor = CurvePreprocessingAgent(
                google_api_key=google_api_key, 
                model_name=model_name, 
                local_model=local_model,
                output_dir=os.path.join(self.output_dir, "preprocessing"),
                executor_timeout=executor_timeout
            )
        else:
            self.preprocessor = None

        if effective_api_key:
            try:
                self.literature_agent = FittingModelLiteratureAgent(api_key=effective_api_key, max_wait_time=max_wait_time)
                logger.info("FittingModelLiteratureAgent initialized successfully.")
            except ValueError as e:
                # Catch the specific error if initialization fails despite finding a key/var
                logger.warning(f"Could not initialize FittingModelLiteratureAgent: {e}. Proceeding without literature search capabilities.")
                self.literature_agent = None
            except Exception as e: # Catch other potential errors during init
                logger.error(f"Unexpected error initializing FittingModelLiteratureAgent: {e}", exc_info=True)
                self.literature_agent = None
        else:
            logger.warning("FutureHouse API key not provided and FUTUREHOUSE_API_KEY env variable not set. Literature search disabled.")
        

    def _load_curve_data(self, data_path: str) -> np.ndarray:
        if data_path.endswith(('.csv', '.txt')):
            data = np.loadtxt(data_path, delimiter=',', skiprows=1)
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Data must be a 2-column array (X, Y).")
        return data

    def _plot_curve(self, curve_data: np.ndarray, system_info: dict, title_suffix="") -> bytes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(curve_data[:, 0], curve_data[:, 1], 'b.', markersize=4)
        plot_title = system_info.get("title")
        if plot_title is None:
            plot_title = "Data"
        ax.set_title(plot_title + title_suffix)
        xlabel_text = system_info.get("xlabel")
        if xlabel_text is None:
            xlabel_text = "X-axis"
        ax.set_xlabel(xlabel_text)
        ylabel_text = system_info.get("ylabel")
        if ylabel_text is None:
            ylabel_text = "Y-axis"
        ax.set_ylabel(ylabel_text)
        ax.grid(True, linestyle='--')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150)
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)
        return image_bytes

    def _generate_lit_search_query(self, plot_bytes: bytes, system_info: dict) -> str:
        """Uses an LLM to formulate a query for the literature agent."""
        self.logger.info("Generating literature search query for fitting models...")
        prompt = [
            LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
            "## Data Plot", {"mime_type": "image/jpeg", "data": plot_bytes},
            self._build_system_info_prompt_section(system_info)
        ]
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result_json, error = self._parse_llm_response(response)
        if error or "search_query" not in result_json:
            raise ValueError("Failed to generate a valid literature search query.")
        return result_json["search_query"]

    def _generate_fitting_script(self, curve_data: np.ndarray, literature_context: str, data_path: str) -> str:
        """Uses an LLM to generate a Python fitting script."""
        self.logger.info("Generating Python script for data fitting...")
        data_preview = np.array2string(curve_data[:10], precision=4, separator=', ')
        prompt = (
            f"{FITTING_SCRIPT_GENERATION_INSTRUCTIONS}\n"
            f"## Literature Context\n{literature_context}\n"
            f"## Curve Data Preview\n{data_preview}\n"
            f"## Data File Path\nThe script should load data from this absolute path: '{os.path.abspath(data_path)}'"
        )
        response = self.model.generate_content(prompt)
        script_content = response.text.strip() # Start by stripping whitespace

        extracted_code = None

        # 1. Try extracting from markdown block (making 'python' optional and case-insensitive)
        match = re.search(r"```(?:python)?\n(.*?)\n```", script_content, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_code = match.group(1).strip()
            self.logger.info("Extracted Python script from markdown block.")
        else:
            # 2. If no markdown, check if it starts with 'python' and strip it
            if script_content.lower().startswith("python"):
                potential_code = script_content[len("python"):].strip() # Remove 'python' prefix
                # Check if what remains looks like Python code
                if potential_code.startswith(("import ", "def ", "#", "'''", '"""')):
                    extracted_code = potential_code
                    self.logger.info("Extracted Python script by stripping 'python' prefix.")
            # 3. If not prefixed with 'python', check if it starts directly with code
            elif script_content.startswith(("import ", "def ", "#", "'''", '"""')):
                extracted_code = script_content
                self.logger.info("Extracted Python script directly (no markdown/prefix).")

        # 4. Check if extraction was successful
        if extracted_code:
             fitting_script = extracted_code
        else:
            # Log the failure and raise
            self.logger.error(f"LLM response did not contain a recognizable Python code block. Response: {script_content[:500]}...")
            raise ValueError("LLM failed to generate Python script in a recognizable format (markdown block, 'python' prefix, or direct code).")

        if not fitting_script: # Should be caught above, but belt-and-suspenders
            raise ValueError("LLM generated an empty or unextractable fitting script.")

        return fitting_script

    def _save_literature_step_results(self, query: str, report: str) -> dict:
        """Saves the literature search query and the resulting report to files."""
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature_step")
            os.makedirs(lit_dir, exist_ok=True)

            # Save the query
            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, 'w') as f:
                f.write(query)
            saved_files["query_file"] = query_path
            self.logger.info(f"Saved literature query to: {query_path}")

            # Save the report
            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            saved_files["report_file"] = report_path
            self.logger.info(f"Saved literature report to: {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to save literature step results: {e}")
        
        return saved_files
    
    def _generate_and_execute_fitting_script_with_retry(
        self, curve_data: np.ndarray, literature_context: str, data_path: str
    ) -> dict:
        """
        Generates and executes the fitting script, with a retry loop for self-correction.
        """
        last_error = "No script generated yet."
        fitting_script = None

        for attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    # First attempt: generate the initial script
                    print(f"⚙️  Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Generating initial fitting script...")
                    fitting_script = self._generate_fitting_script(curve_data, literature_context, data_path)
                else:
                    # Subsequent attempts: generate a corrected script
                    print(f"⚠️  Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Script failed. Requesting correction from LLM...")
                    correction_prompt = FITTING_SCRIPT_CORRECTION_INSTRUCTIONS.format(
                        literature_context=literature_context,
                        failed_script=fitting_script,
                        error_message=last_error
                    )
                    response = self.model.generate_content(correction_prompt)
                    script_content = response.text
                    match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
                    if match:
                        fitting_script = match.group(1).strip()
                    else: # Fallback if markdown block is missing
                        fitting_script = script_content.strip()

                # Execute the current version of the script
                print(f"   Executing script...")
                exec_result = self.executor.execute_script(fitting_script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    print("   ✅ Script executed successfully.")
                    return {
                        "status": "success",
                        "exec_result": exec_result,
                        "final_script": fitting_script,
                        "attempts": attempt
                    }
                else:
                    last_error = exec_result.get("message", "Unknown execution error.")
                    self.logger.warning(f"Script execution attempt {attempt} failed. Error: {last_error}")

            except Exception as e:
                last_error = f"An error occurred during script generation: {str(e)}"
                self.logger.error(last_error, exc_info=True)

        # If loop finishes without success
        print(f"❌ Script generation failed after {self.MAX_SCRIPT_ATTEMPTS} attempts.")
        return {
            "status": "error",
            "message": f"Failed to generate a working script after {self.MAX_SCRIPT_ATTEMPTS} attempts. Last error: {last_error}",
            "last_script": fitting_script
        }
    
    def _evaluate_fit_quality_with_llm(self, original_plot_bytes, fit_plot_bytes, literature_context) -> dict:
        """Uses an LLM to visually assess the quality of the fit."""
        self.logger.info("🤖 Assessing the quality of the curve fit...")
        prompt = [
            FITTING_QUALITY_ASSESSMENT_INSTRUCTIONS,
            "## Original Data Plot", {"mime_type": "image/jpeg", "data": original_plot_bytes},
            "## Fit Visualization", {"mime_type": "image/png", "data": fit_plot_bytes},
            "## Literature Context\n" + literature_context
        ]
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result_json, error = self._parse_llm_response(response)
        
        if error or not result_json:
            self.logger.warning("Failed to get a valid fit quality assessment from LLM. Assuming fit is acceptable.")
            return {"is_good_fit": True, "critique": "Assessment failed.", "suggestion": "N/A"}
            
        return result_json
    
    def _request_model_correction(self, old_script, fit_plot_bytes, critique, suggestion, literature_context) -> str:
        """Asks the LLM to generate a new script with an improved model."""
        self.logger.info("⚠️ Fit was inadequate. Requesting a new model and script from LLM...")
        prompt = [
            FITTING_MODEL_CORRECTION_INSTRUCTIONS,
            "## Critique of Previous Attempt\n" + critique,
            "## Suggestion for Improvement\n" + suggestion,
            "## Plot of the Bad Fit", {"mime_type": "image/png", "data": fit_plot_bytes},
            "## Original Literature Context\n" + literature_context,
            "## Old Script That Produced the Bad Fit\n```python\n" + old_script + "\n```"
        ]
        
        response = self.model.generate_content(prompt)
        script_content = response.text
        
        # ... (extract python code from markdown block as in _generate_fitting_script)
        match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("LLM failed to generate a corrected Python script in a markdown block.")

    def analyze_for_claims(self, data_path: str, system_info: dict = None, **kwargs) -> dict:
        self.logger.info(f"Starting advanced curve analysis with fitting for: {data_path}")

        # This will be the path to the data the fitter *actually* uses.
        # Default to the original input path.
        fit_data_path = data_path
        
        # This is the temp file path, only used if preprocessing runs.
        processed_data_path = os.path.join(self.output_dir, "preprocessed_curve_data.npy") 

        try:
            # Step 0: Load Original Data
            curve_data = self._load_curve_data(data_path)
            system_info = self._handle_system_info(system_info)
            plot_title = " (Raw Data)" # Default plot title

            # --- PREPROCESSING STEP ---
            # Check the flag from __init__
            if self.run_preprocessing and self.preprocessor:
                self.logger.info("--- Running Preprocessing Step ---")
                
                # Run the preprocessing agent
                curve_data, prep_quality = self.preprocessor.run_preprocessing(curve_data, system_info)
                self.logger.info("Preprocessing complete.")
                
                # Save the processed data
                try:
                    np.save(processed_data_path, curve_data)
                    self.logger.info(f"Saved preprocessed data for script to: {processed_data_path}")
                    fit_data_path = processed_data_path # Point the fitter to the new file
                    plot_title = " (Processed Data)" # Update plot title
                except Exception as e:
                    self.logger.error(f"Failed to save temporary processed data: {e}. Fitting will use original data.", exc_info=True)
                    # fit_data_path remains the original data_path
                    # curve_data remains the processed data (for plotting)
            
            else:
                self.logger.info("--- Skipping Preprocessing Step ---")
                # curve_data is the original (raw) data
                # fit_data_path is the original (raw) data_path

            # Step 1: Visualize the data
            # This plots the in-memory curve_data (processed or raw)
            original_plot_bytes = self._plot_curve(curve_data, system_info, plot_title)

            # Initialize fallback values in case literature agent is missing or fails
            lit_query = "N/A (Literature agent not available)"
            literature_context = "Literature agent not available. Using LLM's internal knowledge for model selection."
            saved_lit_files = {}

            # Step 1: Search Literature (Conditional)
            print(f"\n🤖 -------------------- ANALYSIS AGENT STEP: LITERATURE SEARCH -------------------- 🤖")
            if self.literature_agent:
                try:
                    lit_query = self._generate_lit_search_query(original_plot_bytes, system_info)
                    lit_result = self.literature_agent.query_for_models(lit_query)

                    if lit_result["status"] == "success":
                        literature_context = lit_result["formatted_answer"]
                        print("✅ Literature search successful.")
                        saved_lit_files = self._save_literature_step_results(lit_query, literature_context)
                    else:
                        warning_message = f"Literature search failed ({lit_result['message']}). Falling back to LLM's internal knowledge."
                        self.logger.warning(warning_message)
                        print(f"⚠️  {warning_message}")
                        literature_context = "The external literature search failed. Fall back to your internal knowledge to propose a suitable physical fitting model."
                        saved_lit_files = self._save_literature_step_results(lit_query, f"Search Failed: {lit_result['message']}\n\nFallback context:\n{literature_context}")

                except Exception as lit_e:
                    logger.error(f"Error during literature search step: {lit_e}", exc_info=True)
                    print(f"⚠️ Error during literature search: {lit_e}. Falling back to LLM's internal knowledge.")
                    literature_context = "An error occurred during the literature search. Fall back to your internal knowledge to propose a suitable physical fitting model."
                    saved_lit_files = self._save_literature_step_results(lit_query, f"Error during search:\n{lit_e}\n\nFallback context:\n{literature_context}")
            else:
                print("⚠️ Literature agent not available. Using LLM's internal knowledge.")
                saved_lit_files = self._save_literature_step_results(lit_query, literature_context)

            fitting_script = None
            exec_result = None
            fit_plot_bytes = None
            
            for model_attempt in range(1, self.MAX_MODEL_ATTEMPTS + 1):
                print(f"\n🤖 MODELING ATTEMPT {model_attempt}/{self.MAX_MODEL_ATTEMPTS}")
                
                if model_attempt > 1:
                    print(f"\n🤖 --- STEP: GENERATING CORRECTED SCRIPT (from Fit Critique) --- 🤖")
                else:
                    print(f"\n🤖 --- STEP: SCRIPT GENERATION & EXECUTION (from Literature) --- 🤖")
                
                # Pass the correct file path (processed or raw) to the script generator
                script_execution_bundle = self._generate_and_execute_fitting_script_with_retry(
                    curve_data, literature_context, fit_data_path 
                )
                
                if script_execution_bundle["status"] != "success":
                    raise RuntimeError(script_execution_bundle["message"])

                fitting_script = script_execution_bundle["final_script"]
                exec_result = script_execution_bundle["exec_result"]

                # Step 4: Parse Results and Assess Fit Quality
                fit_plot_path = os.path.join(self.output_dir, "fit_visualization.png")
                with open(fit_plot_path, "rb") as f:
                    fit_plot_bytes = f.read()

                assessment = self._evaluate_fit_quality_with_llm(original_plot_bytes, fit_plot_bytes, literature_context)
                print(f"✅ Fit Assessment Critique: {assessment['critique']}")

                is_good_fit_flag = assessment.get("is_good_fit")

                if is_good_fit_flag is True or str(is_good_fit_flag).lower() == 'true':
                    print("✅ Fit quality is acceptable. Proceeding to final interpretation.")
                    break
                
                elif model_attempt < self.MAX_MODEL_ATTEMPTS:
                    print("⚠️ Fit quality is unacceptable. Attempting to correct the model.")
                    literature_context += (
                        f"\n\n--- CRITIQUE OF ATTEMPT {model_attempt} ---\n"
                        f"Critique: {assessment['critique']}\n"
                        f"Suggestion for a better model: {assessment['suggestion']}"
                    )
                else:
                    self.logger.warning("Maximum model attempts reached. Using the last fit despite its imperfections.")
                    print("⚠️ Maximum model attempts reached. Using the last fit despite its imperfections.")
                    break
                
            # Step 5: Final Interpretation (using results from the last successful fit)
            fit_params = {}
            for line in exec_result.get("stdout", "").splitlines():
                if line.startswith("FIT_RESULTS_JSON:"):
                    fit_params = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                    break
            if not fit_params:
                raise ValueError("Could not parse fitting parameters from final script output.")
            
            print(f"\n🤖 -------------------- ANALYSIS AGENT STEP: FINAL INTERPRETATION -------------------- 🤖")
            final_prompt = [
                FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS,
                "\n## Original Data Plot", {"mime_type": "image/jpeg", "data": original_plot_bytes},
                "\n## Final Fit Visualization", {"mime_type": "image/png", "data": fit_plot_bytes},
                "\n## Final Fitted Parameters\n" + json.dumps(fit_params, indent=2),
                "\n## Final Literature Context\n" + literature_context,
                self._build_system_info_prompt_section(system_info)
            ]
            response = self.model.generate_content(final_prompt, generation_config=self.generation_config)
            
            result_json, error_dict = self._parse_llm_response(response)
            if error_dict:
                raise RuntimeError(f"Failed to interpret fitting results: {error_dict}")

            # Store the final plots for the feedback loop
            analysis_images = [
                {'label': 'Original Data Plot', 'data': original_plot_bytes},
                {'label': 'Final Fit Visualization', 'data': fit_plot_bytes}
            ]
            self._store_analysis_images(analysis_images)
            
            initial_result = {
                "detailed_analysis": result_json.get("detailed_analysis"),
                "scientific_claims": self._validate_scientific_claims(result_json.get("scientific_claims", []))
            }
            final_result = self._apply_feedback_if_enabled(initial_result, system_info=system_info)
            
            final_result["status"] = "success"
            final_result["fitting_parameters"] = fit_params
            final_result["literature_files"] = saved_lit_files
            return final_result

        except Exception as e:
            self.logger.exception(f"Curve analysis failed: {e}")
            return {"status": "error", "message": str(e)}

        finally:
            # Always check for the temp file and remove it if it exists.
            if os.path.exists(processed_data_path):
                try:
                    os.remove(processed_data_path)
                    self.logger.debug(f"Cleaned up temporary file: {processed_data_path}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up temp file {processed_data_path}: {e}")

    def _get_claims_instruction_prompt(self) -> str:
        return FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        """
        Returns the instruction prompt for generating measurement recommendations
        based on curve fitting results.
        """
        return CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS