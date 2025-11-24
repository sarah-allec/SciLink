import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import PIL.Image as PIL_Image

from .excel_parser import parse_adaptive_excel
from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS,
    HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK,
    TEA_INSTRUCTIONS_FALLBACK
)


def _parse_json_from_response(resp) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Helper to extract JSON from LLM response objects."""
    if hasattr(resp, 'text'): 
        json_text = resp.text.strip()
    elif hasattr(resp, 'parts') and resp.parts: 
        json_text = resp.parts[0].text.strip()
    else: 
        return None, f"LLM response format unexpected: {resp}"
    
    if json_text.startswith("```json"): 
        json_text = json_text[len("```json"):].strip()
    if json_text.endswith("```"): 
        json_text = json_text[:-len("```")].strip()
        
    try: 
        return json.loads(json_text), None
    except json.JSONDecodeError as e: 
        return None, f"Failed to decode JSON: {str(e)}"


def verify_plan_relevance(objective: str, 
                          result: Dict[str, Any], 
                          model: Any, 
                          generation_config: Any) -> Tuple[bool, str]: 
    """
    Self-reflection step. Returns (True, "") if relevant, or (False, "Reason") if not.
    
    Logic:
    1. Checks if the plan was generated via Fallback (General Knowledge).
    2. If Fallback: Verifies only scientific soundness (Relaxed).
    3. If Strict: Verifies document grounding and specific constraint adherence (Strict).
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments: 
        return False, "No experiments generated."

    # 1. Detect Fallback Mode
    # We check if ANY experiment contains the mandatory fallback warning defined in instruct.py
    is_fallback = False
    for exp in experiments:
        justification = exp.get('justification', '').lower()
        if "general scientific knowledge" in justification or "documents lacked specific context" in justification:
            is_fallback = True
            break

    # 2. Build Plan Summary for the Verifier
    plan_summary_lines = []
    for i, exp in enumerate(experiments):
        name = exp.get('experiment_name', 'N/A')
        hyp = exp.get('hypothesis', 'N/A')
        justification = exp.get('justification', 'No justification provided.')
        
        plan_summary_lines.append(f"Experiment {i+1}: {name}")
        plan_summary_lines.append(f"  Hypothesis: {hyp}")
        plan_summary_lines.append(f"  Justification: {justification}") 
        plan_summary_lines.append("---")
        
    plan_summary = "\n".join(plan_summary_lines)

    # 3. Construct Context-Aware Prompt
    if is_fallback:
        print("    - ℹ️  Verifying Fallback Plan (Relaxed Constraints)...")
        eval_prompt = f"""
        You are a scientific research evaluator.
        
        **CONTEXT:** The system failed to find specific documents for the User Objective in the Knowledge Base.
        Therefore, it generated a plan based on **General Scientific Knowledge**.
        
        1. User Objective: "{objective}"
        2. Proposed Plan (General Knowledge): 
        {plan_summary}

        **TASK:**
        Determine if the Proposed Plan makes scientific sense for the Objective, acknowledging that it CANNOT cite specific documents.
        
        **CRITERIA FOR PASS:**
        - The plan addresses the objective using standard, correct scientific principles.
        - The logic is sound and actionable.
        - **DO NOT FAIL** the plan simply because it uses general knowledge or lacks specific context (this is expected in fallback mode).
        
        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """
    else:
        print("    - ℹ️  Verifying Strict Plan (Document Constraints)...")
        eval_prompt = f"""
        You are a scientific research evaluator.
        
        1. User Objective: "{objective}"
        2. Proposed Plan: 
        {plan_summary}

        **TASK:**
        Review the "Hypothesis" and "Justification" for each experiment.
        Determine if the Proposed Plan is directly relevant to the User Objective AND supported by the cited context.
        
        **CRITERIA FOR FAIL:**
        - The plan ignores specific constraints in the objective (e.g., "Use X method" but the plan uses "Y").
        - The justification contradicts the hypothesis.
        - The plan is logically incoherent.
        
        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """

    # 4. Execute Verification
    try:
        response = model.generate_content([eval_prompt], generation_config=generation_config)
        eval_result, _ = _parse_json_from_response(response)
        
        if eval_result and not eval_result.get("is_relevant"):
            reason = eval_result.get('reason', 'Unknown irrelevance.')
            print(f"    - ⚠️  Plan Verification Failed: {reason}")
            return False, reason
            
        print(f"    - ✅ Plan Verification Passed.")
        return True, ""
        
    except Exception as e:
        logging.error(f"Verification step failed: {e}")
        # Fail open: If the verifier crashes, we assume the plan is okay to avoid blocking the user.
        return True, ""


def perform_science_rag(objective: str, 
                        instructions: str, 
                        task_name: str,
                        kb_docs: Any,  # Pass the KB object here
                        model: Any,    # Pass the LLM object here
                        generation_config: Any,
                        primary_data_set: Optional[Dict[str, str]] = None,
                        image_paths: Optional[List[str]] = None,
                        image_descriptions: Optional[List[str]] = None,
                        additional_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the Scientific/TEA RAG loop using the Docs KnowledgeBase.
    Includes logic for handling Primary Data (Excel) and Fallback generation.
    """
    
    # --- 1. Process Primary Data (e.g., Excel) ---
    primary_data_str = None
    if primary_data_set:
        try:
            chunks = parse_adaptive_excel(primary_data_set['file_path'], primary_data_set['metadata_path'])
            if chunks: 
                summary = next((c for c in chunks if c['metadata'].get('content_type') in ('dataset_summary', 'dataset_package')), chunks[0])
                primary_data_str = summary['text']
        except Exception as e:
            print(f"  - ⚠️ Warning: Failed to parse primary data set: {e}")

    # --- 2. Retrieve Scientific Context (Docs KB Only) ---
    print(f"\n--- Retrieving Scientific Context for {task_name} ---")
    
    doc_chunks = []
    if kb_docs.index and kb_docs.index.ntotal > 0:
        doc_chunks = kb_docs.retrieve(objective, top_k=10)
    
    unique_chunks = {c['text']: c for c in doc_chunks}.values()
    
    if not unique_chunks and not primary_data_str:
        retrieved_context_str = "No specific documents found in Knowledge Base."
    else:
        rag_str = "\n\n---\n\n".join(
            f"Source: {Path(c['metadata'].get('source', 'N/A')).name}\nType: {c['metadata'].get('content_type')}\n\n{c['text']}" 
            for c in unique_chunks
        )
        retrieved_context_str = ""
        if primary_data_str: retrieved_context_str += f"## Primary Data Summary\n{primary_data_str}\n\n"
        if rag_str: retrieved_context_str += f"## Retrieved Scientific Literature\n{rag_str}"

    # --- 3. Construct Multimodal Prompt ---
    loaded_images = []
    img_desc_str = ""
    
    if image_paths and PIL_Image:
        for p in image_paths:
            try: 
                loaded_images.append(PIL_Image.open(p))
            except Exception as e:
                print(f"  - ⚠️ Could not load image {p}: {e}")

    if image_descriptions:
        img_desc_str = json.dumps(image_descriptions, indent=2)

    prompt_parts = [instructions, f"## User Objective:\n{objective}"]
    
    if loaded_images:
        prompt_parts.append("\n## Provided Images: (See attached)")
        prompt_parts.extend(loaded_images)
        if img_desc_str: prompt_parts.append(f"\n## Image Descriptions:\n{img_desc_str}")
    
    if additional_context:
        prompt_parts.append(f"\n## Additional Context:\n{additional_context}")
        
    prompt_parts.append(f"\n## Retrieved Context:\n{retrieved_context_str}")

    # --- 4. Generation & Fallback Logic ---
    print(f"--- Generating {task_name} ---")
    try:
        # Attempt 1: Strict RAG Generation
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        result, error_msg = _parse_json_from_response(response)
        
        if error_msg: 
            return {"error": f"JSON Parsing Error: {error_msg}"}

        # Check for Insufficient Context
        needs_fallback = False
        if result.get("error") and "Insufficient" in str(result.get("error")):
            needs_fallback = True
            print(f"    - ⚠️ Strict generation failed: {result.get('error')}")
        
        # --- 5. Execution of Fallback ---
        if needs_fallback:
            print("    - 🔄 Entering Fallback Mode (General Knowledge)...")
            
            fallback_inst = None
            if instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS:
                fallback_inst = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
            elif instructions == TEA_INSTRUCTIONS:
                fallback_inst = TEA_INSTRUCTIONS_FALLBACK
            
            if not fallback_inst:
                return result # No fallback available for this instruction set

            prompt_parts[0] = fallback_inst
            
            fallback_response = model.generate_content(prompt_parts, generation_config=generation_config)
            result, error_msg_fb = _parse_json_from_response(fallback_response)
            
            if error_msg_fb:
                return {"error": f"Fallback JSON Parsing Error: {error_msg_fb}"}
            
            print("    - ✅ Fallback generation successful.")

        return result

    except Exception as e:
        logging.error(f"Error in perform_science_rag: {e}")
        return {"error": str(e)}


def perform_code_rag(result: Dict[str, Any],
                     kb_code: Any,   # Pass the Code KB object
                     model: Any,     # Pass the LLM object
                     generation_config: Any) -> Dict[str, Any]:
    """
    Retrieves API syntax from the Code KB and generates Python implementation scripts.
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments: 
        return result
    
    # 1. Smart Retrieval: Use the *steps* as the query
    all_steps_text = " ".join([" ".join(e.get('experimental_steps', [])) for e in experiments])
    
    print(f"  - 🔍 Retrieving API syntax for: {all_steps_text[:100]}...")
    hits = kb_code.retrieve(f"python implementation for {all_steps_text}", top_k=5)
    
    if not hits:
        print("    - ℹ️ No relevant code chunks found. Skipping code gen.")
        return result
        
    code_ctx = "\n\n".join([c['text'] for c in hits])
    code_files = list(set([Path(c['metadata']['source']).name for c in hits]))

    # 2. Generate Code
    for exp in experiments:
        steps = exp.get("experimental_steps", [])
        exp_name = exp.get("experiment_name", "Experiment")
        
        prompt = f"""
        You are a Research Software Engineer.
        
        **TASK:** Write a Python script to implement the experimental steps below.
        
        **INPUTS:**
        1. Experimental Steps: {json.dumps(steps)}
        2. API Syntax Reference:
        {code_ctx}
        
        **INSTRUCTIONS:**
        - Use the "API Syntax Reference" to find the correct functions.
        - Map the scientific intent of the Steps to the code.
        - You must prioritize using classes and functions from the API Reference over generic external libraries.
        - Return ONLY valid JSON.
        
        **OUTPUT:** A JSON object: {{ "implementation_code": "YOUR_PYTHON_CODE_HERE" }}
        """
        
        try:
            resp = model.generate_content([prompt], generation_config=generation_config)
            code_res, _ = _parse_json_from_response(resp)
            
            if code_res and "implementation_code" in code_res:
                exp["implementation_code"] = code_res["implementation_code"]
                exp["code_source_files"] = code_files
                print(f"    - ✅ Generated code for '{exp_name}'")
            else:
                print(f"    - ⚠️ Code generation returned no code for '{exp_name}'")
        except Exception as e:
            print(f"    - ❌ Failed to generate code for '{exp_name}': {e}")
            
    return result


def refine_plan_with_feedback(original_result: Dict[str, Any], 
                              feedback: str, 
                              objective: str,
                              model: Any,
                              generation_config: Any) -> Dict[str, Any]:
    """
    Refines the experimental plan based on user input.
    """
    
    refinement_prompt = f"""
    You are an expert Research Strategist acting as an editor.
    
    **Original Objective:** {objective}
    
    **Current Plan (JSON):**
    {json.dumps(original_result, indent=2)}
    
    **User Feedback/Correction:** "{feedback}"
    
    **Task:**
    Update the "Current Plan" to strictly address the "User Feedback".
    
    **Constraints:**
    - You MUST return the exact same JSON structure (keys: "proposed_experiments", etc.).
    - Update "experimental_steps", "hypothesis", or "required_equipment" as requested.
    - Do NOT add explanations outside the JSON.
    
    **Output:**
    A single valid JSON object containing the updated plan.
    """

    try:
        response = model.generate_content([refinement_prompt], generation_config=generation_config)
        refined_result, error_msg = _parse_json_from_response(response)
        
        if error_msg:
            print(f"    - ⚠️ Could not parse refined plan: {error_msg}. Reverting.")
            return original_result
        
        if "proposed_experiments" not in refined_result:
            print("    - ⚠️ Refined plan invalid structure. Reverting.")
            return original_result
            
        return refined_result
        
    except Exception as e:
        print(f"    - ⚠️ Error during refinement: {e}")
        return original_result