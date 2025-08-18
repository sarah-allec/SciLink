import logging
from typing import Dict, Any, List

from .base_agent import BaseAnalysisAgent
from .atomistic_microscopy_agent import AtomisticMicroscopyAnalysisAgent
from .microscopy_agent import MicroscopyAnalysisAgent

from .human_feedback import SimpleFeedbackMixin

HOLISTIC_SYNTHESIS_INSTRUCTIONS = """
You are an expert materials scientist performing a multi-modal synthesis of results from two different analysis methods run on the SAME microscopy image.

You will be given a comprehensive data package for each analysis:
1.  **Atomistic Analysis:**
    - A text summary identifying individual atoms, defects, and local structures.
    - **Analysis Images:** Visual maps showing atomic clustering by intensity, local environment classification, etc.
2.  **General (FFT-NMF) Analysis:**
    - A text summary identifying larger-scale domains and periodicities.
    - **Analysis Images:** Visual maps of NMF components (FFT patterns) and their corresponding abundance maps (spatial locations).

Your task is to act as a senior researcher reviewing all the evidence to formulate a unified analysis.

**Output Format:**
Provide your response in a single JSON object.
{{
  "detailed_analysis": "<Your synthesized, multi-modal analysis text that explicitly references the visual data>",
  "scientific_claims": [
    {{
      "claim": "<A concise scientific claim linking visual evidence from both analyses>",
      "scientific_impact": "<The potential impact of this synthesized finding>",
      "has_anyone_question": "<A 'Has anyone...' question for a literature search>",
      "keywords": ["<keyword1>", "<keyword2>"]
    }}
  ]
}}
"""


class HolisticMicroscopyAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    A composite agent that performs a holistic, multi-modal synthesis of analyses
    from atomistic and general microscopy agents. It uses both text summaries
    and generated analysis images for a comprehensive, multi-scale insight.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Instantiate the specialist agents
        self.atomistic_agent = AtomisticMicroscopyAnalysisAgent(**kwargs)
        
        general_agent_kwargs = kwargs.copy()
        general_agent_kwargs['fft_nmf_settings'] = {
            'FFT_NMF_ENABLED': True,
            'FFT_NMF_AUTO_PARAMS': True,
            'output_dir': 'fft_nmf_output'
        }
        self.general_agent = MicroscopyAnalysisAgent(**general_agent_kwargs)
        self.logger.info("HolisticMicroscopyAgent initialized with specialist sub-agents.")

    def synthesize_analyses(self, atomistic_text: str, general_text: str, 
                            atomistic_images: List[Dict], general_images: List[Dict], 
                            system_info: Dict) -> Dict[str, Any]:
        """
        Takes analysis texts and images from multiple sources and synthesizes them.
        """
        self.logger.info("🧠 Synthesizing multi-modal results with text and images...")
        
        prompt_parts = [HOLISTIC_SYNTHESIS_INSTRUCTIONS]
        
        # Add Atomistic Agent's results
        prompt_parts.append("--- Atomistic Analysis Results ---")
        prompt_parts.append("## Text Summary from Atomistic Agent ##")
        prompt_parts.append(atomistic_text)
        prompt_parts.append("## Visual Evidence from Atomistic Agent ##")
        for img in atomistic_images:
            prompt_parts.append(img.get('label', 'Analysis Image') + ":")
            prompt_parts.append({"mime_type": "image/jpeg", "data": img['data']})

        # Add General Agent's results
        prompt_parts.append("\n--- General (FFT-NMF) Analysis Results ---")
        prompt_parts.append("## Text Summary from General Agent ##")
        prompt_parts.append(general_text)
        prompt_parts.append("## Visual Evidence from General Agent ##")
        for img in general_images:
            prompt_parts.append(img.get('label', 'Analysis Image') + ":")
            prompt_parts.append({"mime_type": "image/jpeg", "data": img['data']})

        # Add system info and final instruction
        prompt_parts.append(self._build_system_info_prompt_section(system_info))
        prompt_parts.append("Synthesize ALL provided information into a final report.")
        
        response = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
        final_json, error = self._parse_llm_response(response)
        
        if error:
            return {"status": "error", "message": "Synthesis failed.", "details": error}
        
        final_json['status'] = 'success'
        self.logger.info("✅ Successfully synthesized holistic results.")
        return final_json

    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: Dict, **kwargs) -> Dict[str, Any]:
        """
        Runs multiple analyses, gathers their text and image outputs, and synthesizes them.
        """
        # 1. Run Atomistic Analysis
        self.logger.info("🔬 Running Atomistic Microscopy Analysis...")
        atomistic_result = self.atomistic_agent.analyze_microscopy_image_for_claims(image_path, system_info)
        atomistic_text = atomistic_result.get("detailed_analysis", "Atomistic analysis failed.")
        atomistic_images = self.atomistic_agent._get_stored_analysis_images()

        # 2. Run General Analysis
        self.logger.info("🗺️ Running General Microscopy Analysis (FFT-NMF)...")
        general_result = self.general_agent.analyze_microscopy_image_for_claims(image_path, system_info)
        general_text = general_result.get("detailed_analysis", "General analysis failed.")
        general_images = self.general_agent._get_stored_analysis_images()

        # 3. Call the synthesis method with all collected data
        synthesized_result = self.synthesize_analyses(
            atomistic_text=atomistic_text,
            general_text=general_text,
            atomistic_images=atomistic_images,
            general_images=general_images,
            system_info=system_info
        )

        if synthesized_result.get('status') == 'success':
            synthesized_result['source_analyses'] = {
                'atomistic_microscopy': atomistic_result,
                'general_microscopy': general_result
            }

        return synthesized_result