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


class IterationFeedbackMixin:
    """Mixin to add intermediate feedback capabilities to iterative agents."""
    
    def __init__(self, *args, enable_human_feedback: bool = False, **kwargs):
        self.enable_human_feedback = enable_human_feedback
        self._feedback_collector = None

    def _collect_and_apply_iteration_feedback(
        self, state: dict, iteration_title: str
    ) -> dict:
        """Collect and apply human feedback specifically for iteration analysis and targets."""
        if not self.enable_human_feedback:
            return state

        collector = SimpleFeedbackCollector(self.logger) # Reuse existing collector functionality

        print("\n" + "="*80)
        print(f"🎯 ITERATION RESULT: {iteration_title}")
        print("="*80)
        
        # Display the LLM's automated decision for the user
        decision = state.get("refinement_decision", {})
        is_needed = decision.get("refinement_needed", False)
        targets = decision.get("targets", [])
        
        print(f"\n🧠 LLM Decision: Refinement Needed = {is_needed}")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        print(f"Targeted Actions ({len(targets)} found):")
        for i, t in enumerate(targets, 1):
            value_str = str(t.get('value', 'N/A'))
            print(f"  {i}. Type: {t.get('type'):<8} | Value: {value_str:<15} | Description: {t.get('description', 'N/A')}")
        
        print("-" * 80)
        
        # Prompt for feedback
        user_feedback = input("\n🤔 Your feedback to adjust the targets/refinement (or press Enter to continue as-is): ").strip()
        
        if not user_feedback:
            print("✅ No feedback provided. Continuing with current targets.")
            return state

        self.logger.info("🔄 Refining targets based on user feedback...")
        
        # Build prompt for refinement (reusing the logic in BaseAnalysisAgent/Controller design)
        prompt_parts = [
            # A new instruction to be created in instruct.py: HYPERSPECTRAL_REFINEMENT_FEEDBACK_INSTRUCTIONS
            # For now, we'll use a placeholder string
            "You are an expert reviewer. The current analysis found the following targets for the next step. A human expert provided feedback. Re-analyze the targets and modify the list (or keep it as is) based on the feedback. You must output the full JSON object (refinement_needed, reasoning, targets).",
            f"\n\n--- CURRENT ITERATION ANALYSIS ---\n{state.get('result_json', {}).get('detailed_analysis', 'N/A')}",
            f"\n\n--- LLM'S ORIGINAL TARGETS ---\n{json.dumps(decision, indent=2)}",
            f"\n\n--- HUMAN FEEDBACK ---\n\"{user_feedback}\"",
            "\n\nProduce the *refined* target selection JSON object (with keys: refinement_needed, reasoning, targets)."
        ]
        
        # Add visual plots for LLM context
        for img in state.get("analysis_images", []):
            if img.get('data'): # Check if image bytes exist
                prompt_parts.append(f"\n{img['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": img['data']})

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            refined_json, error_dict = self._parse_llm_response(response) # Assuming BaseAgent's parser is available
            
            if refined_json and not error_dict:
                # Validate and apply refined decision
                state["refinement_decision"] = {
                    "refinement_needed": refined_json.get("refinement_needed", is_needed),
                    "reasoning": refined_json.get("reasoning", "Refined based on human feedback."),
                    "targets": refined_json.get("targets", targets)
                }
                print("✅ Targets refined successfully.")
            else:
                self.logger.error("LLM failed to refine targets. Using original decision.")

        except Exception as e:
            self.logger.error(f"Error during feedback application: {e}")
            print("❌ Error processing feedback. Using original targets.")
            
        return state