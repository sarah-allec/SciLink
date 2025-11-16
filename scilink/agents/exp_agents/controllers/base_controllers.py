import logging
from typing import Callable

class RunFinalInterpretationController:
    """
    [🧠 LLM Step]
    A generic controller that takes the 'final_prompt_parts' from the state,
    runs the LLM, and stores the 'result_json' and 'error_dict' back in the state.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn  # Pass in the agent's parser

    def execute(self, state: dict) -> dict:
        self.logger.info("🧠 LLM Step: Generating final scientific interpretation...")
        prompt = state.get("final_prompt_parts")
        
        if not prompt:
            self.logger.error("Pipeline reached final step, but no 'final_prompt_parts' in state.")
            state["error_dict"] = {"error": "Pipeline failed to build final prompt"}
            return state
        
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            state["result_json"] = result_json
            state["error_dict"] = error_dict
            
            if not error_dict:
                self.logger.info("✅ LLM Step Complete: Final analysis generated.")
            else:
                self.logger.error(f"❌ LLM Step Failed: {error_dict.get('details')}")

        except Exception as e:
            self.logger.exception(f"❌ LLM Step Failed: {e}")
            state["result_json"] = None
            state["error_dict"] = {"error": "Final LLM analysis failed", "details": str(e)}

        return state

class StoreAnalysisResultsController:
    """
    [🛠️ Tool Step]
    A generic controller that takes the 'analysis_images' from the state
    and saves them using the agent's 'store_fn' for the feedback loop.
    """
    def __init__(self, logger: logging.Logger, store_fn: Callable):
        self.logger = logger
        self._store_analysis_images = store_fn

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Tool Step: Storing analysis images for feedback...")
        
        if state.get("error_dict"):
            self.logger.warning("Skipping storage: An error occurred in the pipeline.")
            return state

        try:
            analysis_metadata = {
                "image_path": state.get("image_path"),
                "system_info": state.get("system_info"),
                "num_stored_images": len(state.get("analysis_images", []))
                # You can add more metadata from the 'state' dict here
            }
            self._store_analysis_images(state.get("analysis_images", []), analysis_metadata)
            self.logger.info("✅ Tool Step Complete: Analysis images stored.")
        except Exception as e:
            self.logger.error(f"❌ Tool Step Failed: Could not store analysis images: {e}")
            
        return state