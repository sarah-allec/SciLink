import logging
import json
from typing import Dict, List, Callable
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from ...auth import get_api_key, APIKeyNotFoundError


PIPELINE_SELECTION_INSTRUCTIONS = """You are an expert materials scientist. 

Your task is to match the input data to one of the pipelines listed above. Use the following logic, which is based on the *purpose* of the pipelines:

* **For Countable Objects:**
    * **Use the pipeline for 'sam' (Segment Anything Model):** This is the correct choice for images containing **large, distinct, countable objects**.
    * **Use-case:** Measuring size distribution, shape, and spatial arrangement of features like nanoparticles, cells, pores, or other discrete entities.

* **For Standard Microstructure:**
    * **Use the 'general' analysis pipeline:** This is for standard microstructure analysis (grains, phases, etc.) where **atoms are not resolved**.

* **For Atomically-Resolved Images (The complex case):**
    You must decide between the 'atomistic' pipeline and the 'general' (FFT/NMF) pipeline.

    * **Use the 'atomistic' pipeline when:**
        * The image is **high-quality** and **individual atoms or atomic columns are clearly visible in a crystalline lattice**.
        * The goal is to analyze well-defined interfaces, grain boundaries, and point defects within an otherwise crystalline structure.

    * **Use the 'general' (FFT/NMF) pipeline when:**
        * The image is dominated by **large-scale disorder**, making direct atom-finding unreliable or less informative.
        * **Examples of such disorder include:**
            * Large amorphous (non-crystalline) regions.
            * Numerous small, disconnected, and poorly-ordered crystalline flakes.
            * Extreme noise levels that obscure the atomic lattice.
        * **For STM images:** Also use this pipeline if the image shows large variations in electronic contrast (LDOS) that are not simple atomic differences, as an FFT-based analysis is more suitable for identifying periodicities.

**Input You Will Receive:**
1. A microscopy image
2. System information (metadata)
3. Optional user-specified analysis goal

You MUST output a valid JSON object with two keys:
1. `pipeline_id`: (String) The ID of the pipeline you have selected
2. `reasoning`: (String) A brief explanation for your choice, justifying it based on the visual data

Output ONLY the JSON object.
"""


class PipelineSelector:
    """
    A general-purpose pipeline selector that uses an LLM to choose
    the most appropriate analysis pipeline based on the input data.
    """
    
    def __init__(self, 
                 google_api_key: str | None = None,
                 model_name: str = "gemini-2.5-flash-preview-05-20",
                 local_model: str = None):
        """
        Initialize the pipeline selector.
        
        Args:
            google_api_key: Google API key for Gemini models
            model_name: Name of the model to use
            local_model: Optional local model endpoint
        """
        self.logger = logging.getLogger(__name__)
        
        # Model initialization (similar to orchestrator)
        if local_model is not None:
            if 'gguf' in local_model:
                self.logger.info(f"💻 Using local agent as pipeline selector.")
                from ...wrappers.llama_wrapper import LocalLlamaModel
                self.model = LocalLlamaModel(local_model)
                self.generation_config = None
                self.safety_settings = None
            elif 'ai-incubator' in local_model:
                self.logger.info(f"🏛️ Using network agent as pipeline selector.")
                from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
                if google_api_key is None:
                    google_api_key = get_api_key('google')
                    if not google_api_key:
                        raise APIKeyNotFoundError('google')
                self.model = OpenAIAsGenerativeModel(model_name, api_key=google_api_key, base_url=local_model)
                self.generation_config = None
                self.safety_settings = None
            else:
                self.logger.info(f"Invalid local_model argument.")
                self.model = None
                self.generation_config = None
                self.safety_settings = None
        else:
            self.logger.info(f"☁️ Using cloud agent as pipeline selector.")
            if google_api_key is None:
                google_api_key = get_api_key('google')
                if not google_api_key:
                    raise APIKeyNotFoundError('google')
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = GenerationConfig(response_mime_type="application/json")
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
    
    def select_pipeline(self,
                       available_pipelines: Dict[str, Dict],
                       image_blob: Dict | None = None,
                       system_info: Dict | None = None) -> tuple[str, str]:
        """
        Select the most appropriate pipeline for the given input.
        
        Args:
            available_pipelines: Dict mapping pipeline_id -> {description, ...}
            image_blob: Optional image data for visual analysis
            system_info: Optional system metadata
            
        Returns:
            tuple: (selected_pipeline_id, reasoning_string)
            Returns (None, error_message) on failure
        """
        self.logger.info("Pipeline selector: Analyzing input to choose best pipeline...")
        
        # Build pipeline descriptions for the prompt
        pipeline_desc_text = ""
        for pid, info in available_pipelines.items():
            pipeline_desc_text += f"- **ID '{pid}'**: {info.get('description', 'No description')}\n"
        
        prompt_parts = [PIPELINE_SELECTION_INSTRUCTIONS.format(pipeline_descriptions=pipeline_desc_text)]
        
        # Add image if available
        if image_blob:
            prompt_parts.append("\n--- Image for Context ---")
            prompt_parts.append("This is the input image to be analyzed:")
            prompt_parts.append(image_blob)
        
        # Add system info
        if system_info:
            prompt_parts.append("\n--- System Information ---")
            prompt_parts.append(json.dumps(system_info, indent=2))
        
        prompt_parts.append("\nBased on the provided information, select the most appropriate pipeline.")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json = self._parse_llm_response(response)
            
            if not result_json:
                return None, "Failed to parse LLM response"
            
            pipeline_id = result_json.get('pipeline_id')
            reasoning = result_json.get('reasoning', 'No reasoning provided.')
            
            self.logger.info(f"\n\n🧠 Pipeline Selector Reasoning: {reasoning}\n")
            
            if pipeline_id not in available_pipelines:
                error_msg = f"LLM selected invalid pipeline: '{pipeline_id}'. Available: {list(available_pipelines.keys())}"
                self.logger.warning(error_msg)
                return None, error_msg
            
            return pipeline_id, reasoning
            
        except Exception as e:
            error_msg = f"Pipeline selection failed: {e}"
            self.logger.exception(error_msg)
            return None, error_msg
    
    def _parse_llm_response(self, response) -> dict | None:
        """Parse JSON response from LLM."""
        try:
            raw_text = response.text
            first_brace = raw_text.find('{')
            last_brace = raw_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_string = raw_text[first_brace:last_brace + 1]
                return json.loads(json_string)
            else:
                raise ValueError("Could not find valid JSON in response.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM JSON response: {e}")
            return None