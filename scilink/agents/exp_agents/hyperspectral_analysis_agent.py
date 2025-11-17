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
from .pipelines.hyperspectral_pipelines import create_hyperspectral_pipeline
from ...tools.image_processor import load_image, convert_numpy_to_jpeg_bytes # For structure image

class HyperspectralAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Refactored agent for analyzing hyperspectral data using a modular,
    controller-based pipeline.
    """

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

        # --- Pipeline Initialization ---
        self.pipeline = create_hyperspectral_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.spectral_settings,
            preprocessor=self.preprocessor, # Inject the dependency
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images
        )
        self.logger.info(f"HyperspectralAnalysisAgent initialized with a pipeline of {len(self.pipeline)} controllers.")

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
        The agent's main execution engine.
        It prepares the initial state and runs the loaded pipeline.
        """
        try:
            # --- 1. Common State Initialization ---
            self.logger.info(f"--- Starting analysis pipeline for {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            hspy_data = self._load_hyperspectral_data(data_path)

            # Handle optional structure image
            structure_image_blob = None
            if structure_image_path and os.path.exists(structure_image_path):
                try:
                    img = load_image(structure_image_path)
                    # Use simple normalization for structure img, not full preprocess
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    structure_image_blob = {
                        "mime_type": "image/jpeg",
                        "data": convert_numpy_to_jpeg_bytes(img)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not load structure image {structure_image_path}: {e}")
            
            # This is the "state" dictionary that is passed through the pipeline
            state = {
                "data_path": data_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "hspy_data": hspy_data, # The raw data
                "settings": self.spectral_settings, # Pass settings to controllers
                
                # Optional structure context
                "structure_image_path": structure_image_path,
                "structure_system_info": self._handle_system_info(structure_system_info),
                "structure_image_blob": structure_image_blob,
                
                # Fields to be filled by the pipeline
                "analysis_images": [], # Will be filled by controllers
                "result_json": None,
                "error_dict": None
            }
            
            # --- 2. Run the Pipeline ---
            for controller in self.pipeline:
                state = controller.execute(state)
                # Stop pipeline if a major error occurred
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}. Stopping execution.")
                    break

            # --- 3. Return Final Results (from the state) ---
            self.logger.info(f"--- Analysis pipeline finished. ---")
            return state.get("result_json"), state.get("error_dict")

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
            instruction_prompt=SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
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
            instruction_prompt=SPECTROSCOPY_ANALYSIS_INSTRUCTIONS,
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
        return SPECTROSCOPY_CLAIMS_INSTRUCTIONS

    def _get_measurement_recommendations_prompt(self) -> str:
        return SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS