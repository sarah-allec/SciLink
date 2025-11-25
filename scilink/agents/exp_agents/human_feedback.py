import json
import logging
from typing import Dict, Callable, Any, Optional
from datetime import datetime

from google.generativeai.types import GenerationConfig

from .instruct import ITERATION_REFINEMENT_INSTRUCTIONS 


class SimpleFeedbackCollector:
    """Simple feedback collector for refining analysis results"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def collect_optional_feedback(self, analysis_result: Dict[str, Any]) -> Optional[str]:
        """Intuitive feedback collection that handles natural user behavior."""
        detailed_analysis = analysis_result.get("detailed_analysis", "No analysis provided")
        claims = analysis_result.get("scientific_claims", [])
        
        print("\n" + "="*80)
        print("🤖 AGENT's ANALYSIS RESULTS")
        print("="*80)
        
        # Display detailed analysis
        print("\n📋 DETAILED ANALYSIS:")
        print("-" * 40)
        print(detailed_analysis)
        
        # Display claims
        print(f"\n🎯 SCIENTIFIC CLAIMS ({len(claims)} generated):")
        print("-" * 40)
        
        if not claims:
            print("❌ No claims were generated.")
        else:
            for i, claim in enumerate(claims, 1):
                print(f"\nCLAIM {i}:")
                print(f"  Statement: {claim.get('claim', 'N/A')}")
                print(f"  Impact: {claim.get('scientific_impact', 'N/A')}")
                print(f"  Research Question: {claim.get('has_anyone_question', 'N/A')}")
                print(f"  Keywords: {', '.join(claim.get('keywords', []))}")
        
        print("\n" + "="*80)
        
        try:
            # Single, clear prompt that accepts feedback directly
            feedback_text = input("\n🤔 Your feedback (or press Enter to use analysis as-is): ").strip()
            
            if not feedback_text:
                print("✅ No feedback provided. Using analysis as-is.")
                return None
            
            print(f"\n✅ Feedback collected ({len(feedback_text)} characters)")
            return feedback_text
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Using original analysis.")
            return None
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
            return None
    
    def save_feedback_log(self, original_result: Dict[str, Any], 
                         feedback: str, 
                         refined_result: Dict[str, Any] = None,
                         output_dir: str = "feedback_logs") -> str:
        """Save feedback session for audit trail."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"feedback_log_{timestamp}.json")
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "original_result": original_result,
            "user_feedback": feedback,
            "refined_result": refined_result,
            "session_summary": {
                "feedback_provided": feedback is not None,
                "feedback_length": len(feedback) if feedback else 0,
                "original_claims_count": len(original_result.get("scientific_claims", [])),
                "refined_claims_count": len(refined_result.get("scientific_claims", [])) if refined_result else 0
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Feedback log saved: {log_file}")
        return log_file


class SimpleFeedbackMixin:
    """Mixin to add feedback capabilities to existing analysis agents."""
    
    def __init__(self, *args, enable_human_feedback: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_human_feedback = enable_human_feedback
        self._feedback_collector = None
    
    def _get_feedback_collector(self) -> SimpleFeedbackCollector:
        """Lazy initialization of feedback collector."""
        if self._feedback_collector is None:
            self._feedback_collector = SimpleFeedbackCollector(self.logger)
        return self._feedback_collector
    
    def _apply_feedback_if_enabled(self, claims_result: Dict[str, Any], 
                                  **refinement_kwargs) -> Dict[str, Any]:
        """Apply human feedback to refine analysis if enabled."""
        if not self.enable_human_feedback:
            return claims_result
        
        if "error" in claims_result:
            return claims_result
        
        try:
            feedback_collector = self._get_feedback_collector()
            
            # Collect optional feedback
            user_feedback = feedback_collector.collect_optional_feedback(claims_result)
            
            if not user_feedback:
                return claims_result
            
            self.logger.info("🔄 Refining analysis based on user feedback...")
            
            # Use the common refinement method
            refined_result = self._refine_with_feedback(
                claims_result, user_feedback, **refinement_kwargs
            )
            
            if "error" in refined_result:
                self.logger.error("Refinement failed, using original analysis")
                return claims_result
            
            # Add feedback metadata
            refined_result["human_feedback"] = {
                "feedback_provided": True,
                "user_feedback": user_feedback,
                "refinement_applied": True
            }
            
            # Save feedback log
            feedback_collector.save_feedback_log(
                claims_result, user_feedback, refined_result
            )
            
            print("✅ Analysis refined based on your feedback!")
            return refined_result
            
        except Exception as e:
            self.logger.error(f"Feedback processing failed: {e}")
            return claims_result
    
    def _refine_with_feedback(self, original_result: Dict[str, Any], 
                            user_feedback: str, **kwargs) -> Dict[str, Any]:
        """
        Common refinement method for all agents.
        Uses stored images and agent-specific instruction prompt.
        """
        try:
            # Get stored images from the original analysis
            stored_images = self._get_stored_analysis_images()
            
            if not stored_images:
                self.logger.warning("No stored images available for refinement")
                return {"error": "No stored analysis images available for refinement"}
            
            self.logger.info(f"Using {len(stored_images)} stored analysis images for refinement")
            
            # Get agent-specific instruction prompt
            instruction_prompt = self._get_claims_instruction_prompt()
            
            # Use base class refinement with stored images
            return self._refine_analysis_with_feedback(
                original_analysis=original_result.get("detailed_analysis", ""),
                original_claims=original_result.get("scientific_claims", []),
                user_feedback=user_feedback,
                instruction_prompt=instruction_prompt,
                stored_images=stored_images,
                system_info=kwargs.get('system_info')
            )
            
        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return {"error": "Refinement failed", "details": str(e)}
    
    def _get_claims_instruction_prompt(self) -> str:
        """
        Return the instruction prompt for claims analysis.
        Must be implemented by each agent.
        """
        raise NotImplementedError("Each agent must implement _get_claims_instruction_prompt()")


class IterativeFeedbackController:
    """
    [🧠 LLM/User Step] Facilitates human-in-the-loop validation and refinement 
    of LLM-generated analysis plans (e.g., refinement targets).
    
    It requests LLM refinement based on human feedback on a structured decision JSON.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.refinement_instruction = ITERATION_REFINEMENT_INSTRUCTIONS 

    def execute(self, state: dict) -> dict:
        # Check if human feedback is globally enabled (via agent settings)
        # Note: 'enable_human_feedback' must be passed into the 'settings' key in state.
        if not state.get('settings', {}).get('enable_human_feedback', False):
             self.logger.info("Feedback skipped: Human feedback not enabled for this agent.")
             return state

        decision = state.get("refinement_decision")
        if not decision:
            self.logger.warning("Feedback skipped: 'refinement_decision' missing from state.")
            return state

        self.logger.info("\n\n👤 --- USER STEP: REVIEW ANALYSIS PLAN --- 👤\n")
        
        # --- 1. Display Current Decision to User and Collect Feedback ---
        iteration_title = state.get("iteration_title", "Current Analysis")
        analysis_text = state.get("result_json", {}).get("detailed_analysis", "No analysis text provided.")
        targets = decision.get("targets", [])
        
        print("\n" + "="*80)
        print(f"🎯 ANALYSIS STEP REVIEW: {iteration_title}")
        print("="*80)
        print("\n**SUMMARY OF CURRENT ANALYSIS:**")
        print(analysis_text)
        print("-" * 80)
        
        print(f"🧠 LLM's Proposed Plan: Refinement Needed = **{decision.get('refinement_needed', False)}**")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        print(f"\nTargeted Actions ({len(targets)} found):")
        
        for i, t in enumerate(targets, 1):
            value_str = str(t.get('value', 'N/A'))
            print(f"  {i}. Type: {t.get('type'):<15} | Value: {value_str:<15} | Description: {t.get('description', 'N/A')}")
        
        print("-" * 80)
        
        try:
            user_feedback = input("\n🤔 Your feedback to adjust the targets/plan (or press Enter to accept): ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted feedback. Accepting original decision.")
            return state

        if not user_feedback:
            self.logger.info("✅ User accepted original refinement decision.")
            return state
        
        # --- 3. Run LLM Refinement with Full Context ---
        self.logger.info("🔄 Refining decision using full scientific context...")
        
        # Prepare context for the LLM
        system_info_json = json.dumps(state.get("system_info", {}), indent=2)

        prompt_parts = [
            f"You are an expert reviewer. Use the HUMAN EXPERT FEEDBACK to produce a **REVISED** and definitive version of the analysis plan JSON object. The human input overrides the initial automated logic.",
            
            f"\n\n--- LLM'S ORIGINAL DECISION JSON ---\n{json.dumps(decision, indent=2)}",
            f"\n\n--- CURRENT ITERATION'S DETAILED ANALYSIS ---\n{analysis_text}",
            f"\n\n--- CURRENT SYSTEM METADATA ---\n{system_info_json}",
            f"\n\n--- HUMAN EXPERT FEEDBACK ---\n\"{user_feedback}\"",
            "\n\n--- VISUAL CONTEXT (Plots from Current Analysis) ---\n"
        ]

        # Add all analysis images for visual context
        for img in state.get("analysis_images", []):
            image_bytes = img.get('data') or img.get('bytes') 
            if image_bytes:
                prompt_parts.append(f"\n{img['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
        
        # Append the refinement instruction (which defines the output JSON structure)
        prompt_parts.append(f"\n\nProvide your final, REVISED plan strictly adhering to the format defined in the instruction below. {self.refinement_instruction}")
        
        # Call LLM for structured revision
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            refined_json, error_dict = self._parse_llm_response(response)

            if refined_json and not error_dict:
                state["refinement_decision"] = refined_json
                self.logger.info(f"✅ Refinement success. Final decision: {refined_json.get('reasoning', 'No reasoning').strip()}")
                print(f"\n✅ REFINED: New plan established based on feedback.")
            else:
                self.logger.error("❌ LLM failed to produce a valid refinement JSON. Retaining original decision.")
                print("\n❌ Refinement failed due to bad LLM output. Retaining original plan.")

        except Exception as e:
            self.logger.error(f"❌ Error during LLM refinement call: {e}")
            print("\n❌ Critical error during refinement. Retaining original plan.")
            
        return state