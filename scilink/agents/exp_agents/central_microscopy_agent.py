from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent
from .human_feedback import SimpleFeedbackMixin
from .pipeline_selector import PipelineSelector
from .pipeline_registry import (
    get_available_pipelines,
    create_pipeline_for_agent,
    get_prompt_for_pipeline
)

from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)


class CentralMicroscopyAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """    
    Unified microscopy agent that can use multiple analysis pipelines.
    
    The agent uses an LLM-based pipeline selector to automatically choose
    the most appropriate pipeline based on the input image and metadata:
    - 'general': FFT/NMF analysis for standard microstructures
    - 'sam': Particle segmentation for countable objects
    - 'atomistic': Atomic-resolution analysis for crystalline materials
    
    agent_setting configuration:
    --------------
    - auto_select_pipeline (bool): If True (default), use LLM to select pipeline.
                                    If False, use 'default_pipeline' setting.
    - default_pipeline (str): Pipeline to use when auto_select_pipeline=False.
                              Options: 'general', 'sam', 'atomistic'
    - pipeline_settings (dict): Settings specific to each pipeline, keyed by pipeline_id
    
    Example configuration:
    {
        'auto_select_pipeline': True,  # Let LLM choose
        'default_pipeline': 'general',  # Fallback if auto-selection fails
        'pipeline_settings': {
            'general': {'FFT_NMF_ENABLED': True, ...},
            'sam': {'refinement_cycles': 1, ...},
            'atomistic': {'refine_positions': True, ...}
        }
    }
    
    Backward Compatibility:
    ----------------------
    The old individual parameters (fft_nmf_settings, sam_settings, atomistic_analysis_settings)
    are still supported and will be automatically converted to the new format.
    """

    def __init__(self,
                 google_api_key: str | None = None,
                 model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 agent_settings: dict | None = None,
                 enable_human_feedback: bool = True,
                 selector_model_name="gemini-2.5-flash-preview-05-20",
                 # Backward compatibility parameters
                 fft_nmf_settings: dict | None = None,
                 sam_settings: dict | None = None,
                 atomistic_analysis_settings: dict | None = None):
        
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        
        # Handle backward compatibility
        if agent_settings is None:
            agent_settings = self._build_legacy_settings(
                fft_nmf_settings, sam_settings, atomistic_analysis_settings
            )
        
        self.agent_settings = agent_settings
        self._recommendation_agent = None
        
        # Initialize pipeline selector
        self.auto_select = agent_settings.get('auto_select_pipeline', True)
        if self.auto_select:
            self.pipeline_selector = PipelineSelector(
                google_api_key=google_api_key,
                model_name=selector_model_name,
                local_model=local_model
            )
        else:
            self.pipeline_selector = None
        
        # Store the default pipeline ID
        self.default_pipeline_id = agent_settings.get('default_pipeline', 'general')
        
        # These will be set during analysis
        self.current_pipeline = None
        self.current_pipeline_id = None
        
        self.logger.info(f"MicroscopyAnalysisAgent initialized. Auto-select: {self.auto_select}, Default: {self.default_pipeline_id}")

    def _build_legacy_settings(self, fft_nmf_settings, sam_settings, atomistic_settings) -> dict:
        """Convert old-style settings to new unified format."""
        agent_settings = {
            'auto_select_pipeline': True,
            'default_pipeline': 'general',
            'pipeline_settings': {}
        }
        
        if fft_nmf_settings:
            agent_settings['pipeline_settings']['general'] = fft_nmf_settings
            agent_settings['default_pipeline'] = 'general'
            self.logger.info("Converted legacy fft_nmf_settings to new format")
        
        if sam_settings:
            agent_settings['pipeline_settings']['sam'] = sam_settings
            if not fft_nmf_settings:  # Only change default if general wasn't set
                agent_settings['default_pipeline'] = 'sam'
            self.logger.info("Converted legacy sam_settings to new format")
        
        if atomistic_settings:
            agent_settings['pipeline_settings']['atomistic'] = atomistic_settings
            if not fft_nmf_settings and not sam_settings:  # Only change default if others weren't set
                agent_settings['default_pipeline'] = 'atomistic'
            self.logger.info("Converted legacy atomistic_analysis_settings to new format")
        
        return agent_settings

    def _select_and_create_pipeline(self, image_blob: dict, system_info: dict) -> tuple[list, str, str]:
        """
        Select and create the appropriate pipeline for analysis.
        
        Returns:
            tuple: (pipeline, pipeline_id, reasoning)
        """
        available_pipelines = get_available_pipelines('microscopy')
        
        # Auto-select pipeline if enabled
        if self.auto_select and self.pipeline_selector:
            pipeline_id, reasoning = self.pipeline_selector.select_pipeline(
                available_pipelines=available_pipelines,
                image_blob=image_blob,
                system_info=system_info
            )
            
            if pipeline_id is None:
                self.logger.warning(f"Pipeline selection failed: {reasoning}. Using default: {self.default_pipeline_id}")
                pipeline_id = self.default_pipeline_id
                reasoning = f"Using default pipeline due to selection failure: {reasoning}"
        else:
            pipeline_id = self.default_pipeline_id
            reasoning = f"Auto-selection disabled. Using configured default pipeline: {pipeline_id}"
            self.logger.info(reasoning)
        
        # Get pipeline-specific settings
        pipeline_settings = self.agent_settings.get('pipeline_settings', {}).get(pipeline_id, {})
        
        # Create the pipeline
        self.logger.info(f"Creating pipeline '{pipeline_id}' with settings: {pipeline_settings}")
        
        pipeline = create_pipeline_for_agent(
            pipeline_id=pipeline_id,
            agent_type='microscopy',
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=pipeline_settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
        )
        
        return pipeline, pipeline_id, reasoning

    def _run_analysis_pipeline(
        self, 
        image_path: str, 
        system_info: dict, 
        prompt_type: str = 'claims',
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        The agent's main execution engine.
        It selects the appropriate pipeline, prepares the initial state, and runs it.
        
        Args:
            image_path: Path to the image file
            system_info: Metadata dictionary
            prompt_type: Type of prompt ('analysis', 'claims', 'recommendations')
            additional_context: Optional additional context for the prompt
        """
        try:
            # --- 1. Common State Initialization ---
            self.logger.info(f"--- Starting analysis pipeline for {image_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            loaded_image = load_image(image_path)
            nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(system_info, loaded_image.shape)
            
            preprocessed_img_array, _ = preprocess_image(loaded_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}
            
            # --- 2. Pipeline Selection ---
            pipeline, pipeline_id, selection_reasoning = self._select_and_create_pipeline(
                image_blob, system_info
            )
            
            # Store for later use
            self.current_pipeline = pipeline
            self.current_pipeline_id = pipeline_id
            
            # Display selection reasoning to user
            print("\n" + "="*80)
            print("🔀 PIPELINE SELECTION")
            print(f"  Selected Pipeline: '{pipeline_id}'")
            print(f"  Reasoning: {selection_reasoning}")
            print("="*80 + "\n")
            
            # Get the appropriate instruction prompt for this pipeline
            instruction_prompt = get_prompt_for_pipeline(pipeline_id, 'microscopy', prompt_type)
            
            # --- 3. Create Initial State ---
            state = {
                "image_path": image_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "additional_top_level_context": additional_context,
                "image_blob": image_blob,
                "preprocessed_image_array": preprocessed_img_array,
                "nm_per_pixel": nm_per_pixel,
                "fov_in_nm": fov_in_nm,
                "analysis_images": [
                    {"label": "Primary Microscopy Image", "data": image_bytes}
                ],
                "result_json": None,
                "error_dict": None,
                "settings": self.agent_settings.get('pipeline_settings', {}).get(pipeline_id, {})
            }
            
            # Add atomistic-specific state if needed
            if pipeline_id == 'atomistic':
                # Get model path using the tool function
                from ...tools.atomistic_model_manager import get_or_download_atomistic_model
                model_dir_path = get_or_download_atomistic_model(state["settings"], self.logger)
                if not model_dir_path:
                    return None, {"error": "DCNN model directory not available"}
                state["model_dir_path"] = model_dir_path
                
                # Rescale image if needed
                if fov_in_nm is not None:
                    from ...tools import atomistic_tools
                    rescaled_image, _, final_pixel_size_A = atomistic_tools.rescale_for_model(
                        preprocessed_img_array, fov_in_nm
                    )
                    state["preprocessed_image_array"] = rescaled_image
                    state["nm_per_pixel"] = final_pixel_size_A / 10.0
                    self.logger.info(f"Image rescaled. New pixel size: {state['nm_per_pixel']*10:.3f} Å/px.")

            # --- 4. Run the Pipeline ---
            for controller in pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}. Stopping execution.")
                    break

            # --- 5. Return Final Results ---
            self.logger.info(f"--- Analysis pipeline finished. ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred during the analysis pipeline: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}

    # --- Public API Methods ---

    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Analyze microscopy image to generate DFT structure recommendations.
        """
        # Text-Only Path
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")
            if not self._recommendation_agent:
                self._recommendation_agent = RecommendationAgent(self.google_api_key, self.model_name, self.local_model)
            return self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )
        
        # Image-Based Path
        elif image_path:
            self.logger.info("Generating DFT recommendations via selected pipeline.")
            result_json, error_dict = self._run_analysis_pipeline(
                image_path, 
                system_info, 
                prompt_type='analysis',  # Use 'analysis' prompt for recommendations
                additional_context=additional_prompt_context
            )
            
            if error_dict: return error_dict
            if result_json is None: return {"error": "Analysis failed unexpectedly."}

            recommendations = result_json.get("structure_recommendations", [])
            sorted_recs = self._validate_structure_recommendations(recommendations)
            
            if not sorted_recs:
                self.logger.warning("Pipeline ran but LLM returned no valid recommendations.")

            return {
                "analysis_summary_or_reasoning": result_json.get("detailed_analysis", "Analysis complete, but no text was returned."), 
                "recommendations": sorted_recs
            }
        
        else:
            return {"error": "Either image_path or (cached_detailed_analysis...) must be provided."}

    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims.
        """
        result_json, error_dict = self._run_analysis_pipeline(
            image_path, 
            system_info, 
            prompt_type='claims'
        )

        if error_dict: return error_dict
        if result_json is None: return {"error": "Analysis for claims failed unexpectedly."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        if not valid_claims:
            self.logger.warning("Pipeline ran but LLM returned no valid claims.")
            
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis complete, but no text was returned."), 
            "scientific_claims": valid_claims
        }
        
        return self._apply_feedback_if_enabled(
            initial_result, 
            image_path=image_path, 
            system_info=system_info
        )
    
    def _get_claims_instruction_prompt(self) -> str:
        """Return the appropriate claims prompt for the current pipeline."""
        if self.current_pipeline_id:
            return get_prompt_for_pipeline(self.current_pipeline_id, 'microscopy', 'claims')
        # Fallback
        return get_prompt_for_pipeline('general', 'microscopy', 'claims')
    
    def _get_measurement_recommendations_prompt(self) -> str:
        """Return the appropriate recommendations prompt for the current pipeline."""
        if self.current_pipeline_id:
            return get_prompt_for_pipeline(self.current_pipeline_id, 'microscopy', 'recommendations')
        # Fallback
        return get_prompt_for_pipeline('general', 'microscopy', 'recommendations')