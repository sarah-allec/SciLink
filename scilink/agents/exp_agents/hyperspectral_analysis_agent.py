import os
import numpy as np
import cv2
from typing import Dict, Any

from .base_agent import BaseAnalysisAgent
from .instruct import (
    SPECTROSCOPY_ANALYSIS_INSTRUCTIONS,
    SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
    SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .human_feedback import SimpleFeedbackMixin
from .preprocess import HyperspectralPreprocessingAgent
from .pipelines.hyperspectral_pipelines import (
    create_hyperspectral_iteration_pipeline, 
    create_hyperspectral_synthesis_pipeline
)
from ...tools.image_processor import load_image, convert_numpy_to_jpeg_bytes # For structure image

class HyperspectralAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Refactored agent for analyzing hyperspectral data using a modular,
    controller-based pipeline.
    
    This agent now implements a recursive "survey-then-focus" loop.
    It runs an analysis, uses an LLM to select a region to "zoom in" on,
    and re-runs the analysis on that subset. It continues this loop
    until no further refinement is needed, then synthesizes all results.
    """
    
    MAX_REFINEMENT_ITERATIONS = 4 # Global + 3 zoom-ins

    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 spectral_unmixing_settings: dict | None = None,
                 run_preprocessing: bool = True,
                 output_dir: str = "spectroscopy_output",
                 enable_human_feedback: bool = False):
        
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        
        # --- Settings ---
        default_settings = {
            'method': 'nmf',
            'n_components': 4, # Default if auto_components=False
            'normalize': True,
            'enabled': True,
            'auto_components': True,
            'min_auto_components': 2,
            'max_auto_components': 8
        }
        self.spectral_settings = spectral_unmixing_settings if spectral_unmixing_settings else default_settings
        self.spectral_settings['run_preprocessing'] = run_preprocessing
        self.spectral_settings['output_dir'] = output_dir
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # --- Sub-Agent Initialization ---
        # The preprocessor is a dependency required by the pipeline
        self.preprocessor = HyperspectralPreprocessingAgent(
            google_api_key=google_api_key,
            model_name=model_name,
            local_model=local_model
        )

        # --- Common Pipeline Arguments ---
        pipeline_args = {
            "model": self.model,
            "logger": self.logger,
            "generation_config": self.generation_config,
            "safety_settings": self.safety_settings,
            "settings": self.spectral_settings,
            "parse_fn": self._parse_llm_response,
        }

        # --- Pipeline Initialization ---
        self.iteration_pipeline = create_hyperspectral_iteration_pipeline(
            **pipeline_args,
            preprocessor=self.preprocessor # Iteration pipeline needs this
        )
        self.synthesis_pipeline = create_hyperspectral_synthesis_pipeline(
            **pipeline_args,
            store_fn=self._store_analysis_images # Only synthesis pipeline stores
        )
        self.logger.info(f"HyperspectralAnalysisAgent initialized with recursive pipelines.")

    def _load_hyperspectral_data(self, data_path: str) -> np.ndarray:
        """
        Load hyperspectral data from numpy array.
        Assumes data_path points to a .npy file.
        """
        try:
            if not data_path.endswith('.npy'):
                raise ValueError(f"Expected .npy file, got: {data_path}")
            
            data = np.load(data_path)
            self.logger.info(f"Loaded hyperspectral data with shape: {data.shape}")
            
            if data.ndim == 2:
                self.logger.warning("2D data detected, assuming single spectrum. Reshaping to (1, 1, n_channels)")
                data = data.reshape(1, 1, -1)
            elif data.ndim != 3:
                raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
                
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load hyperspectral data from {data_path}: {e}")
            raise

    def _run_analysis_pipeline(
        self,
        data_path: str,
        system_info: dict,
        instruction_prompt: str,
        structure_image_path: str | None = None,
        structure_system_info: dict | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        The agent's main execution engine, now a recursive loop.
        It prepares an initial state and runs the iteration pipeline,
        storing results and re-running on a subset until told to stop.
        It then runs a final synthesis pipeline on all collected results.
        """
        try:
            # --- 1. Initial State Initialization ---
            self.logger.info(f"--- Starting RECURSIVE analysis pipeline for {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            original_hspy_data = self._load_hyperspectral_data(data_path)

            # Handle optional structure image
            structure_image_blob = None
            if structure_image_path and os.path.exists(structure_image_path):
                try:
                    img = load_image(structure_image_path)
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    structure_image_blob = {
                        "mime_type": "image/jpeg",
                        "data": convert_numpy_to_jpeg_bytes(img)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not load structure image {structure_image_path}: {e}")
            
            # --- 2. Recursive Loop Setup ---
            all_iteration_results = []
            iteration_count = 0
            
            # This is the base state shared by all iterations
            base_state = {
                "data_path": data_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt, # Used by synthesis
                "settings": self.spectral_settings,
                "original_hspy_data": original_hspy_data, # Unmodified data
                "structure_image_path": structure_image_path,
                "structure_system_info": self._handle_system_info(structure_system_info),
                "structure_image_blob": structure_image_blob,
                "continue_loop": True,
                "data_for_this_iteration": original_hspy_data, # Data to be sliced
                "iteration_title": "Global Analysis",
                "error_dict": None
            }

            current_state = base_state.copy()

            while iteration_count < self.MAX_REFINEMENT_ITERATIONS and current_state.get("continue_loop", False):
                self.logger.info(f"\n--- STARTING HYPERSPECTRAL ITERATION {iteration_count} ({current_state.get('iteration_title', '')}) ---\n")
                
                # Create a fresh state for this iteration
                iteration_state = current_state.copy()
                iteration_state["analysis_images"] = [] # Reset images for this iteration
                iteration_state["hspy_data"] = iteration_state["data_for_this_iteration"]
                
                # Disable preprocessing after the first run
                if iteration_count > 0:
                    iteration_state["settings"] = iteration_state["settings"].copy()
                    iteration_state["settings"]['run_preprocessing'] = False
                
                # --- 3. Run Iteration Pipeline ---
                for controller in self.iteration_pipeline:
                    iteration_state = controller.execute(iteration_state)
                    if iteration_state.get("error_dict"):
                        self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}. Stopping loop.")
                        current_state["continue_loop"] = False
                        break
                
                if iteration_state.get("error_dict"):
                    break # Exit loop on error
                
                # --- 4. Store Iteration Results ---
                all_iteration_results.append({
                    "iteration_title": iteration_state.get('iteration_title', f'Iteration {iteration_count}'),
                    "iteration_analysis_text": iteration_state.get("result_json", {}).get("detailed_analysis", "Analysis text not found."),
                    "analysis_images": iteration_state.get("analysis_images", []),
                    "refinement_decision": iteration_state.get("refinement_decision", {}),
                    "final_components": iteration_state.get("final_components"),
                    "final_abundance_maps": iteration_state.get("final_abundance_maps")
                })

                # --- 5. Update Loop State for Next Iteration ---
                current_state["continue_loop"] = iteration_state.get("continue_loop", False)
                if current_state["continue_loop"]:
                    # Prepare data for next loop
                    current_state["data_for_this_iteration"] = iteration_state.get("data_for_this_iteration")
                    current_state["iteration_title"] = iteration_state.get("iteration_title")
                    current_state["system_info"] = iteration_state.get("system_info")
                    # Extract reasoning from the decision that triggered this continuation
                    decision = iteration_state.get("refinement_decision", {})
                    if decision.get("refinement_needed"):
                        current_state["parent_refinement_reasoning"] = decision.get("reasoning", "Refinement requested.")
                    else:
                        current_state["parent_refinement_reasoning"] = None
                iteration_count += 1
            
            # --- 6. Run Final Synthesis ---
            self.logger.info(f"\n--- RECURSIVE LOOP FINISHED. RUNNING FINAL SYNTHESIS. ({len(all_iteration_results)} iterations) ---\n")

            synthesis_state = base_state.copy()
            synthesis_state["all_iteration_results"] = all_iteration_results
            synthesis_state["instruction_prompt"] = instruction_prompt # Pass the *original* prompt

            for controller in self.synthesis_pipeline:
                synthesis_state = controller.execute(synthesis_state)
                if synthesis_state.get("error_dict"):
                    self.logger.error(f"Synthesis pipeline failed at step {controller.__class__.__name__}.")
                    break

            # --- 7. Return Final Results ---
            self.logger.info(f"--- Analysis pipeline finished. ---")
            return synthesis_state.get("result_json"), synthesis_state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Hyperspectral data file not found: {data_path}")
            return None, {"error": "Hyperspectral data file not found", "details": f"Path: {data_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred during the analysis pipeline: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}

    # --- Public API Methods ---

    def analyze_for_claims(self, data_path: str, metadata_path: Dict[str, Any] | str | None = None,
                           structure_image_path: str = None, structure_system_info: Dict[str, Any] = None
                           ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data to generate scientific claims.
        """
        result_json, error_dict = self._run_analysis_pipeline(
            data_path=data_path,
            system_info=metadata_path,
            instruction_prompt=SPECTROSCOPY_CLAIMS_INSTRUCTIONS, # This will be used by the *synthesis* controller
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info
        )
        
        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "Spectroscopy analysis for claims failed unexpectedly."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis not provided."),
            "scientific_claims": valid_claims
        }
        
        return self._apply_feedback_if_enabled(
            initial_result,
            system_info=self._handle_system_info(metadata_path) # Ensure it's a dict for feedback
        )
        
    def analyze_hyperspectral_data(self, data_path: str, metadata_path: str,
                                   structure_image_path: str = None,
                                   structure_system_info: Dict[str, Any] = None
                                   ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data for materials characterization (standard analysis).
        """
        result_json, error_dict = self._run_analysis_pipeline(
            data_path=data_path,
            system_info=metadata_path,
            instruction_prompt=SPECTROSCOPY_ANALYSIS_INSTRUCTIONS, # Used by synthesis
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info
        )
        
        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "Spectroscopy analysis failed unexpectedly."}

        self.logger.info("Hyperspectral analysis completed successfully")
        return result_json

    def _get_claims_instruction_prompt(self) -> str:
        # This is now used by the *feedback* mechanism.
        # The main prompts are passed directly in _run_analysis_pipeline.
        return SPECTROSCOPY_CLAIMS_INSTRUCTIONS

    def _get_measurement_recommendations_prompt(self) -> str:
        return SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS