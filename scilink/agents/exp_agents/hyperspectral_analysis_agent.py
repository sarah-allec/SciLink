import os
import numpy as np
import cv2
from typing import Dict, Any
from collections import deque

from .base_agent import BaseAnalysisAgent
from .instruct import (
    SPECTROSCOPY_ANALYSIS_INSTRUCTIONS,
    SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
    SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .human_feedback import SimpleFeedbackMixin, IterationFeedbackMixin
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
    
    MAX_REFINEMENT_ITERATIONS = 2 # Global + 2 zoom-ins

    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 spectral_unmixing_settings: dict | None = None,
                 run_preprocessing: bool = True,
                 output_dir: str = "spectroscopy_output",
                 enable_human_feedback: bool = False):
        
        BaseAnalysisAgent.__init__(self, google_api_key, model_name, local_model)
        SimpleFeedbackMixin.__init__(self, enable_human_feedback=enable_human_feedback)
        
        # --- Settings ---
        default_settings = {
            'method': 'nmf',
            'n_components': 4, # Default if auto_components=False
            'normalize': True,
            'enabled': True,
            'auto_components': True,
            'min_auto_components': 2,
            'max_auto_components': 8,
            'enable_human_feedback': enable_human_feedback
        }
        self.spectral_settings = spectral_unmixing_settings if spectral_unmixing_settings else default_settings
        self.spectral_settings['run_preprocessing'] = run_preprocessing
        self.spectral_settings['output_dir'] = output_dir
        self.spectral_settings['feedback_depths'] = [0]
        
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
        The agent's main execution engine using a Queue-Based Branching architecture.
        
        It starts with a Global Analysis task. If the LLM identifies distinct features 
        (e.g., "Phase A" and "Phase B"), it generates new tasks for each, adding them 
        to the queue. The agent processes tasks until the queue is empty or max depth is reached.
        """
        try:
            # --- 1. Initial State Initialization ---
            self.logger.info(f"--- Starting BRANCHING analysis pipeline for {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            # Load the raw hyperspectral data
            original_hspy_data = self._load_hyperspectral_data(data_path)

            # Handle optional structure image (Loaded once, passed to all tasks)
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
            
            # --- 2. Initialize the Task Queue ---
            # Define the root task (Global Analysis)
            initial_task = {
                "data": original_hspy_data,           # The data chunk to analyze
                "system_info": system_info,           # Metadata specific to this chunk
                "title": "Global_Analysis",           # Display title
                "parent_reasoning": None,             # Context: Why are we looking at this?
                "depth": 0                            # Recursion depth
            }
            
            # Use a deque for efficient popping from the left (Breadth-First)
            task_queue = deque([initial_task])
            
            # Storage for the results of every completed task
            all_completed_results = []
            
            # --- 3. The Processing Loop ---
            task_counter = 0
            
            while task_queue:
                # Get the next task
                current_task = task_queue.popleft() 
                task_counter += 1
                
                # Guardrail: Max depth check
                if current_task["depth"] > self.MAX_REFINEMENT_ITERATIONS:
                    self.logger.info(f"Skipping task '{current_task['title']}': Max recursion depth ({self.MAX_REFINEMENT_ITERATIONS}) reached.")
                    continue

                self.logger.info(f"\n=== PROCESSING TASK {task_counter}: {current_task['title']} (Depth {current_task['depth']}) ===\n")

                # Construct the State for this specific task
                # This isolates the data/context for this specific iteration
                iteration_state = {
                    "data_path": data_path,
                    "hspy_data": current_task["data"],               # The specific slice for this task
                    "original_hspy_data": original_hspy_data,        # Reference for spectral slicing tools
                    "system_info": current_task["system_info"],      # Specific metadata (e.g. sliced energy range)
                    "instruction_prompt": instruction_prompt,        # Base instructions
                    "settings": self.spectral_settings.copy(),       # Settings (copied to allow modification)
                    "iteration_title": current_task["title"],
                    "parent_refinement_reasoning": current_task["parent_reasoning"],
                    "current_depth": current_task["depth"],
                    
                    # Structure image context (passed to all tasks)
                    "structure_image_path": structure_image_path,
                    "structure_system_info": self._handle_system_info(structure_system_info),
                    "structure_image_blob": structure_image_blob,
                    
                    # Containers for results
                    "analysis_images": [],
                    "error_dict": None
                }

                # Optimization: Only run heavy preprocessing (despike/mask) on the Global task (Depth 0).
                # Sub-tasks act on data that is already processed/sliced.
                if current_task["depth"] > 0:
                    iteration_state["settings"]['run_preprocessing'] = False

                # --- Run the Iteration Pipeline ---
                # This runs NMF, visualization, and the Refinement Decision Logic
                for controller in self.iteration_pipeline:
                    iteration_state = controller.execute(iteration_state)
                    
                    # Critical Error Check
                    if iteration_state.get("error_dict"):
                        self.logger.error(f"Pipeline failed at step {controller.__class__.__name__} for task '{current_task['title']}'.")
                        break
                
                # If the pipeline failed effectively, skip saving results (or save partial error results)
                if iteration_state.get("error_dict"):
                    continue

                # --- Store Results ---
                # Save the summary and images for the Final Synthesis
                result_summary = {
                    "iteration_title": iteration_state.get("iteration_title"),
                    "iteration_analysis_text": iteration_state.get("result_json", {}).get("detailed_analysis", "Analysis text not found."),
                    "analysis_images": iteration_state.get("analysis_images", []),
                    "refinement_decision": iteration_state.get("refinement_decision", {}),
                    "depth": current_task["depth"],
                    "custom_analysis_metadata": iteration_state.get("custom_analysis_metadata")
                }
                all_completed_results.append(result_summary)

                # --- Process New Branches ---
                # The `GenerateRefinementTasksController` populates `new_tasks` based on LLM targets
                new_tasks = iteration_state.get("new_tasks", [])
                
                if new_tasks:
                    self.logger.info(f"--> Task '{current_task['title']}' spawned {len(new_tasks)} new sub-tasks.")
                    for t in new_tasks:
                        # Map the controller output to our queue format
                        queue_item = {
                            "data": t["data"],
                            "system_info": t["system_info"],
                            "title": t["title"],
                            "parent_reasoning": t["parent_reasoning"],
                            "depth": t["source_depth"]
                        }
                        task_queue.append(queue_item)

            # --- 4. Run Final Synthesis ---
            self.logger.info(f"\n=== QUEUE EMPTY. Synthesizing {len(all_completed_results)} analyses ===\n")
            
            # Create state for the synthesis pipeline
            synthesis_state = {
                "all_iteration_results": all_completed_results,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "result_json": None,
                "error_dict": None
            }

            for controller in self.synthesis_pipeline:
                synthesis_state = controller.execute(synthesis_state)
                if synthesis_state.get("error_dict"):
                    self.logger.error(f"Synthesis pipeline failed at step {controller.__class__.__name__}.")
                    break

            # --- 5. Return Final Results ---
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


    def analyze_for_claims(self, data_path: str, metadata_path: Dict[str, Any] | str | None = None,
                           structure_image_path: str = None, structure_system_info: Dict[str, Any] = None
                           ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data to generate scientific claims.
        """
        # 1. Run the Pipeline (Generates Draft 1 Report)
        result_json, error_dict = self._run_analysis_pipeline(
            data_path=data_path,
            system_info=metadata_path,
            instruction_prompt=SPECTROSCOPY_CLAIMS_INSTRUCTIONS, 
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info
        )
        
        if error_dict: return error_dict
        if result_json is None: return {"error": "Spectroscopy analysis failed unexpectedly."}

        # 2. Get Valid Claims (Draft 1)
        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis not provided."),
            "scientific_claims": valid_claims
        }
        
        # 3. Apply Feedback (Generates Draft 2 Text)
        final_result = self._apply_feedback_if_enabled(
            initial_result,
            system_info=self._handle_system_info(metadata_path)
        )

        # 4. Check if feedback changed the result. If so, regenerate the HTML report.
        if self.enable_human_feedback and final_result != initial_result:
            self.logger.info("🔄 Feedback applied. Regenerating HTML report with refined analysis...")
            
            # Reconstruct the state required by the Controller
            # We fetch images from the BaseAgent's storage
            stored_images = self._get_stored_analysis_images() 
            
            repot_state = {
                "result_json": final_result,  # Use the REFINED text
                "system_info": self._handle_system_info(metadata_path),
                "analysis_images": stored_images,
                "image_path": data_path
            }
            
            # Manually instantiate and run the controller
            from .controllers.hyperspectral_controllers import GenerateHTMLReportController
            report_gen = GenerateHTMLReportController(self.logger, self.spectral_settings)
            report_gen.execute(repot_state)
            
            self.logger.info("✅ Refined HTML report generated.")
        
        return final_result
        
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