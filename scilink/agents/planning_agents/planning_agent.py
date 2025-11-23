import google.generativeai as genai
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import PIL.Image
PIL_Image = PIL.Image

from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass, chunk_text
from .excel_parser import parse_adaptive_excel
from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS, 
    HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK, 
    TEA_INSTRUCTIONS_FALLBACK
)
from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel

class PlanningAgent:
    """
    Orchestrates RAG pipelines using explicitly separated Knowledge Bases
    for Scientific Context and Implementation Code.
    """
    def __init__(self, google_api_key: str = None,
                 model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 embedding_model: str = "gemini-embedding-001",
                 kb_base_path: str = "./kb_storage/default_kb",
                 code_chunk_size: int = 5000): 
        
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')

        # --- LLM Backend Configuration ---
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  Using OpenAI-compatible model for generation: {model_name}")
            self.model = OpenAIAsGenerativeModel(model_name, api_key=google_api_key, base_url=local_model)
            self.generation_config = None
        else:
            logging.info(f"☁️  Using Google Gemini model for generation: {model_name}")
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        self.code_chunk_size = code_chunk_size

        # --- Dual KnowledgeBase Initialization ---
        base_path = Path(kb_base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Scientific/Docs KB
        self.kb_docs = KnowledgeBase(google_api_key=google_api_key, embedding_model=embedding_model, local_model=local_model)
        self.kb_docs_prefix = base_path.parent / f"{base_path.name}_docs"
        self.kb_docs_index = str(self.kb_docs_prefix.with_suffix(".faiss"))
        self.kb_docs_chunks = str(self.kb_docs_prefix.with_suffix(".json"))

        # 2. Implementation/Code KB
        self.kb_code = KnowledgeBase(google_api_key=google_api_key, embedding_model=embedding_model, local_model=local_model)
        self.kb_code_prefix = base_path.parent / f"{base_path.name}_code"
        self.kb_code_index = str(self.kb_code_prefix.with_suffix(".faiss"))
        self.kb_code_chunks = str(self.kb_code_prefix.with_suffix(".json"))

        print("--- Initializing Agent (Dual-KB System) ---")
        self._load_knowledge_bases()

    def _load_knowledge_bases(self):
        """Attempts to load both KBs from disk."""
        print(f"  - Docs KB: Loading from {self.kb_docs_prefix}...")
        docs_loaded = self.kb_docs.load(self.kb_docs_index, self.kb_docs_chunks)
        
        print(f"  - Code KB: Loading from {self.kb_code_prefix}...")
        code_loaded = self.kb_code.load(self.kb_code_index, self.kb_code_chunks)

        self._kb_is_built = docs_loaded or code_loaded
        
        if docs_loaded: print("    - ✅ Docs KB loaded.")
        if code_loaded: print("    - ✅ Code KB loaded.")
        if not self._kb_is_built: print("    - ⚠️  No pre-built KBs found.")

    def _parse_json_from_response(self, resp) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if hasattr(resp, 'text'): json_text = resp.text.strip()
        elif hasattr(resp, 'parts') and resp.parts: json_text = resp.parts[0].text.strip()
        else: return None, f"LLM response format unexpected: {resp}"
        
        if json_text.startswith("```json"): json_text = json_text[len("```json"):].strip()
        if json_text.endswith("```"): json_text = json_text[:-len("```")].strip()
        try: return json.loads(json_text), None
        except json.JSONDecodeError as e: return None, f"Failed to decode JSON: {str(e)}"

    def _save_results_to_json(self, results: Dict[str, Any], file_path: str):
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            print(f"    - ✅ Results successfully saved to: {file_path}")
        except Exception as e: logging.error(f"    - ❌ Failed to save results: {e}")

    def _process_file_list(self, file_paths: List[str], is_code_mode: bool) -> List[Dict[str, Any]]:
        """
        Generic helper to process a list of files.
        If is_code_mode=True, treats text files as code blocks and tags metadata as 'code'.
        """
        chunks = []
        for f_path in file_paths:
            path = Path(f_path)
            if not path.exists():
                print(f"  - ⚠️ File not found: {f_path}")
                continue
                
            file_ext = path.suffix.lower()
            
            # PDF Processing (Used for both Science papers and PDF Manuals)
            if file_ext == '.pdf':
                pdf_chunks = extract_pdf_two_pass(f_path)
                # If this PDF is being added to the Code KB, force the content_type to code
                if is_code_mode:
                    for c in pdf_chunks: c['metadata']['content_type'] = 'code'
                chunks.extend(pdf_chunks)
            
            # Text/Code Processing
            elif file_ext in ['.txt', '.md', '.py', '.java', '.r', '.cpp', '.h', '.js', '.json', '.csv']:
                try:
                    with path.open('r', encoding='utf-8') as f: content = f.read()
                    
                    # Format: if code mode, wrap in code blocks. If science mode, treat as text.
                    if is_code_mode:
                        formatted_text = f"CODE FILE: {path.name}\n\n```\n{content}\n```"
                        chunk_sz = self.code_chunk_size
                        ctype = 'code'
                    else:
                        formatted_text = f"DOCUMENT: {path.name}\n\n{content}"
                        chunk_sz = 1000 # Standard text size
                        ctype = 'text'

                    new_chunks = chunk_text(formatted_text, page_num=1, chunk_size=chunk_sz, overlap=50)
                    for c in new_chunks: 
                        c['metadata']['content_type'] = ctype
                        c['metadata']['source'] = f_path
                    chunks.extend(new_chunks)
                    print(f"  - Extracted {len(new_chunks)} chunks from {path.name} ({'Code' if is_code_mode else 'Docs'} Mode)")
                except Exception as e:
                    print(f"  - ❌ Error reading {f_path}: {e}")
            else:
                print(f"  - ⚠️ Unsupported file type: {f_path}")
        return chunks

    def _build_and_save_kb(self,
                           science_paths: Optional[List[str]] = None,
                           code_paths: Optional[List[str]] = None,
                           structured_data_sets: Optional[List[Dict[str, str]]] = None
                           ) -> bool:
        """
        Builds TWO separate knowledge bases based on explicit input lists.
        """
        print("\n--- Rebuilding Knowledge Bases ---")
        
        # 1. Build Docs KB (Science)
        doc_chunks = []
        if science_paths:
            print(f"Processing {len(science_paths)} Scientific Documents...")
            doc_chunks.extend(self._process_file_list(science_paths, is_code_mode=False))
        
        if structured_data_sets:
            print(f"Processing {len(structured_data_sets)} Structured Data Sets...")
            for data_set in structured_data_sets:
                try:
                    if Path(data_set['file_path']).suffix.lower() in ['.xlsx', '.xls']:
                        excel_chunks = parse_adaptive_excel(data_set['file_path'], data_set['metadata_path'])
                        if excel_chunks: doc_chunks.extend(excel_chunks)
                except Exception as e: print(f"  - ❌ Error processing Excel: {e}")

        if doc_chunks:
            print(f"  - Building Scientific KB with {len(doc_chunks)} chunks...")
            self.kb_docs.build(doc_chunks)
            self.kb_docs.save(self.kb_docs_index, self.kb_docs_chunks)
        else:
            print("  - ℹ️  No Scientific docs provided. Docs KB unchanged (or empty).")

        # 2. Build Code KB (Implementation)
        code_chunks = []
        if code_paths:
            print(f"Processing {len(code_paths)} Implementation/Code Documents...")
            code_chunks.extend(self._process_file_list(code_paths, is_code_mode=True))
        
        if code_chunks:
            print(f"  - Building Code KB with {len(code_chunks)} chunks...")
            self.kb_code.build(code_chunks)
            self.kb_code.save(self.kb_code_index, self.kb_code_chunks)
        else:
            print("  - ℹ️  No Code docs provided. Code KB unchanged (or empty).")

        self._kb_is_built = True
        print("✅ Dual-KB Build Complete.")
        return True

    def _ensure_kb_is_ready(self, science_paths, code_paths, structured_data_sets) -> bool:
        new_inputs = (science_paths or []) or (code_paths or []) or (structured_data_sets or [])
        if new_inputs:
            return self._build_and_save_kb(science_paths, code_paths, structured_data_sets)
        elif not self._kb_is_built:
            logging.error("Knowledge base is not built.")
            return False
        return True

    def _verify_plan_relevance(self, objective: str, result: Dict[str, Any]) -> bool:
        experiments = result.get("proposed_experiments", [])
        if not experiments: return False

        plan_summary_lines = []
        for i, exp in enumerate(experiments):
            name = exp.get('experiment_name', 'N/A')
            hyp = exp.get('hypothesis', 'N/A')
            # Extract the justification so the verifier knows "WHY"
            justification = exp.get('justification', 'No justification provided.')
            
            plan_summary_lines.append(f"Experiment {i+1}: {name}")
            plan_summary_lines.append(f"  Hypothesis: {hyp}")
            plan_summary_lines.append(f"  Justification: {justification}") 
            plan_summary_lines.append("---")
            
        plan_summary = "\n".join(plan_summary_lines)

        eval_prompt = f"""
        You are a scientific research evaluator.
        
        1. User Objective: "{objective}"
        2. Proposed Plan: 
        {plan_summary}

        **Task:**
        Review the "Hypothesis" and "Justification" for each experiment and determine if the Proposed Plan is directly relevant to the User Objective.
        
        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """

        try:
            response = self.model.generate_content([eval_prompt], generation_config=self.generation_config)
            eval_result, _ = self._parse_json_from_response(response)
            
            if eval_result and not eval_result.get("is_relevant"):
                print(f"    - ⚠️  Plan Verification Failed: {eval_result.get('reason')}")
                return False
                
            print(f"    - ✅ Plan Verification Passed.")
            return True
            
        except Exception as e:
            logging.error(f"Verification step failed: {e}")
            return True

    def _perform_science_rag(self, objective: str, instructions: str, task_name: str, 
                           primary_data_set: Optional[Dict[str, str]] = None,
                           image_paths: Optional[List[str]] = None,
                           image_descriptions: Optional[List[str]] = None,
                           additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the RAG loop for Scientific Planning or TEA.
        
        Logic Flow:
        1. Parse Primary Data (Excel) if available.
        2. Retrieve Scientific Context (Docs KB only).
        3. Generate Response.
        4. IF response indicates "Insufficient Context":
           - Switch to Fallback Instructions (General Knowledge).
           - Re-generate.
        """
        
        # --- 1. Process Primary Data (e.g., Excel) ---
        primary_data_str = None
        if primary_data_set:
            try:
                # We use the adaptive parser to get a summary of the dataset
                chunks = parse_adaptive_excel(primary_data_set['file_path'], primary_data_set['metadata_path'])
                if chunks: 
                    # Prefer the summary chunk, fallback to the first chunk
                    summary = next((c for c in chunks if c['metadata'].get('content_type') in ('dataset_summary', 'dataset_package')), chunks[0])
                    primary_data_str = summary['text']
            except Exception as e:
                print(f"  - ⚠️ Warning: Failed to parse primary data set: {e}")

        # --- 2. Retrieve Scientific Context (Docs KB Only) ---
        print(f"\n--- Retrieving Scientific Context for {task_name} ---")
        
        doc_chunks = []
        if self.kb_docs.index and self.kb_docs.index.ntotal > 0:
            # We specifically DO NOT query kb_code here. 
            # We want pure scientific context to form the hypothesis first.
            doc_chunks = self.kb_docs.retrieve(objective, top_k=10)
        
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
        
        # Load Images if PIL is available
        if image_paths and PIL_Image:
            for p in image_paths:
                try: 
                    loaded_images.append(PIL_Image.open(p))
                except Exception as e:
                    print(f"  - ⚠️ Could not load image {p}: {e}")

        if image_descriptions:
            img_desc_str = json.dumps(image_descriptions, indent=2)

        # Build the Prompt List (Text + Images)
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
            response = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
            result, error_msg = self._parse_json_from_response(response)
            
            if error_msg: 
                return {"error": f"JSON Parsing Error: {error_msg}"}

            # Check if the LLM refused due to lack of context
            # (The strict instructions tell it to return an "error" key if context is missing)
            needs_fallback = False
            if result.get("error") and "Insufficient" in str(result.get("error")):
                needs_fallback = True
                print(f"    - ⚠️ Strict generation failed: {result.get('error')}")
            
            # --- 5. Execution of Fallback ---
            if needs_fallback:
                print("    - 🔄 Entering Fallback Mode (General Knowledge)...")
                
                # Determine which fallback instruction set to use
                if instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS:
                    fallback_inst = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
                elif instructions == TEA_INSTRUCTIONS:
                    fallback_inst = TEA_INSTRUCTIONS_FALLBACK
                else:
                    # If we don't have a specific fallback, return the error
                    return result

                # Update the instructions in the prompt (Index 0)
                prompt_parts[0] = fallback_inst
                
                # Attempt 2: General Knowledge Generation
                fallback_response = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
                result, error_msg_fb = self._parse_json_from_response(fallback_response)
                
                if error_msg_fb:
                    return {"error": f"Fallback JSON Parsing Error: {error_msg_fb}"}
                
                print("    - ✅ Fallback generation successful.")

            return result

        except Exception as e:
            logging.error(f"Error in _perform_science_rag: {e}")
            return {"error": str(e)}

    def propose_experiments(self, objective: str, 
                            science_paths: Optional[List[str]] = None, 
                            code_paths: Optional[List[str]] = None,
                            structured_data_sets: Optional[List[Dict[str, str]]] = None,
                            tea_summary: Optional[str] = None,
                            primary_data_set: Optional[Dict[str, str]] = None,
                            image_paths: Optional[List[str]] = None,
                            image_descriptions: Optional[List[str]] = None,
                            output_json_path: Optional[str] = None,
                            enable_human_feedback: bool = True):
        """
        Generates experimental plans using a Phased Architecture:
        Phase 1: Science (Logic & Steps) - Uses Docs KB
        Phase 2: Feedback (Human Loop) - Refines Logic
        Phase 3: Engineering (Code) - Uses Code KB
        """
        
        # --- Init Knowledge Bases ---
        if not self._ensure_kb_is_ready(science_paths, code_paths, structured_data_sets):
            return {"error": "KB Init Failed"}

        # =====================================================
        # PHASE 1: SCIENCE STRATEGY (Docs KB Only)
        # =====================================================
        print(f"\n--- Phase 1: Generating Experimental Strategy ---")
        
        ctx = f"TEA Findings:\n{tea_summary}" if tea_summary else None
        
        # This generates the Plan, Steps, and Hypothesis (No Code)
        res = self._perform_science_rag(
            objective, HYPOTHESIS_GENERATION_INSTRUCTIONS, "Experimental Plan", 
            primary_data_set, image_paths, image_descriptions, ctx)
                
        # =====================================================
        # PHASE 2: HUMAN FEEDBACK LOOP
        # =====================================================
        if enable_human_feedback and res.get("proposed_experiments") and not res.get("error"):
            
            # Show the Science Plan
            self._display_plan_summary(res)
            
            # Capture user input
            user_feedback = self._get_user_feedback()
            
            if user_feedback:
                print(f"\n📝 Feedback received. Refining Scientific Plan...")
                res = self._refine_plan_with_feedback(res, user_feedback, objective)
                self._display_plan_summary(res)
                print("✅ Scientific plan updated.")
            else:
                print("✅ Scientific plan accepted.")

        # =====================================================
        # PHASE 3: CODE IMPLEMENTATION (Code KB Only)
        # =====================================================
        # Now we generate code for the *Finalized* steps
        if self.kb_code.index and self.kb_code.index.ntotal > 0 and not res.get("error"):
             print(f"\n--- Phase 3: Mapping to Implementation Code ---")
             res = self._perform_code_rag(res)

        # --- Save & Return ---
        if output_json_path: self._save_results_to_json(res, output_json_path)
        return res

    def perform_technoeconomic_analysis(self, objective: str,
                                        science_paths: Optional[List[str]] = None,
                                        code_paths: Optional[List[str]] = None, 
                                        structured_data_sets: Optional[List[Dict[str, str]]] = None,
                                        primary_data_set: Optional[Dict[str, str]] = None,
                                        image_paths: Optional[List[str]] = None,
                                        image_descriptions: Optional[List[str]] = None,
                                        output_json_path: Optional[str] = None):
        """Performs TEA using Dual-KB retrieval."""
        
        if not self._ensure_kb_is_ready(science_paths, code_paths, structured_data_sets):
            return {"error": "KB Init Failed"}

        res = self._perform_science_rag(objective, TEA_INSTRUCTIONS, "Technoeconomic Analysis",
                                        primary_data_set, image_paths, image_descriptions)
        if output_json_path: self._save_results_to_json(res, output_json_path)
        return res
    
    def save_implementation_scripts(self, result_json: Dict[str, Any], base_output_dir: str):
        """
        Reads the LLM's structured JSON output and saves any generated 
        'implementation_code' strings as separate .py files.
        """
        if result_json.get("error"):
            print("  - ⚠️ Skipping code script saving: LLM output contained an error.")
            return

        experiments = result_json.get("proposed_experiments", [])
        if not experiments:
            print("  - ⚠️ Skipping code script saving: No experiments found in JSON output.")
            return

        base_path = Path(base_output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        num_saved = 0

        print(f"\n--- Saving Generated Code Scripts to: {base_output_dir} ---")

        for i, exp in enumerate(experiments):
            code_content = exp.get("implementation_code")
            exp_name = exp.get("experiment_name", f"Experiment_{i+1}")
            
            # Clean filename
            safe_name = "".join(c for c in exp_name if c.isalnum() or c in (' ', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            if not safe_name: safe_name = f"experiment_code_{i+1}"
            
            filename = f"{safe_name}.py"
            file_path = base_path / filename

            if code_content and "No relevant code found" not in code_content:
                try:
                    # Strip markdown ```
                    code_lines = code_content.splitlines()
                    start_index = next((j for j, line in enumerate(code_lines) if line.strip().startswith('```')), -1)
                    end_index = next((j for j, line in enumerate(code_lines[start_index+1:]) if line.strip().endswith('```')), -1)
                    
                    if start_index != -1 and end_index != -1:
                        extracted_code = "\n".join(code_lines[start_index + 1 : end_index + start_index + 1]).strip()
                    else:
                        extracted_code = code_content.strip()

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_code)
                    
                    print(f"  - ✅ Saved script for '{exp_name}' to: {file_path.name}")
                    num_saved += 1
                except Exception as e:
                    logging.error(f"  - ❌ Failed to save code script {filename}: {e}")
            else:
                 print(f"  - ℹ️ Experiment {i+1} has no executable code content. Skipping save.")

        if num_saved > 0:
            print(f"--- Successfully saved {num_saved} executable script(s). ---")
        else:
             print("--- No executable scripts were generated or saved. ---")

    def _perform_code_rag(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified method to generate code. 
        Used for Initial, Fallback, and Refined plans.
        """
        experiments = result.get("proposed_experiments", [])
        if not experiments: return result
        
        # 1. Smart Retrieval: Use the *steps* as the query
        all_steps_text = " ".join([" ".join(e.get('experimental_steps', [])) for e in experiments])
        
        print(f"  - 🔍 Retrieving API syntax for: {all_steps_text[:100]}...")
        hits = self.kb_code.retrieve(f"python implementation for {all_steps_text}", top_k=5)
        
        if not hits:
            print("    - ℹ️ No relevant code chunks found. Skipping code gen.")
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
            - Return ONLY valid JSON.
            
            **OUTPUT:** A JSON object: {{ "implementation_code": "YOUR_PYTHON_CODE_HERE" }}
            """
            
            try:
                # We use a fresh generation call so the context isn't polluted
                resp = self.model.generate_content([prompt], generation_config=self.generation_config)
                code_res, _ = self._parse_json_from_response(resp)
                
                if code_res and "implementation_code" in code_res:
                    exp["implementation_code"] = code_res["implementation_code"]
                    exp["code_source_files"] = code_files
                    print(f"    - ✅ Generated code for '{exp_name}'")
                else:
                    print(f"    - ⚠️ Code generation returned no code for '{exp_name}'")
            except Exception as e:
                print(f"    - ❌ Failed to generate code for '{exp_name}': {e}")
                
        return result

    # -------------------------------------------------------------------------
    # 4. New Helper: User Feedback Input
    # -------------------------------------------------------------------------
    def _get_user_feedback(self) -> Optional[str]:
        """
        Pauses execution to get user input. 
        Returns None if the user just presses ENTER.
        """
        print("\n" + "-"*60)
        
        print("👤 HUMAN FEEDBACK STEP")
        print("-" * 60)
        print("Review the plan above.")
        print("• To APPROVE: Press [ENTER] directly.")
        print("• To REQUEST CHANGES: Type your feedback/instructions and press [ENTER].")
        
        feedback = input("\n> Instruction: ").strip()
        
        if not feedback:
            return None # User accepted the plan
        return feedback
    

    def _refine_plan_with_feedback(self, original_result: Dict[str, Any], 
                               feedback: str, objective: str) -> Dict[str, Any]:
        """
        Refines the experimental plan using the LLM based on user input.
        Strictly enforces JSON structure preservation.
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
            response = self.model.generate_content([refinement_prompt], generation_config=self.generation_config)
            refined_result, error_msg = self._parse_json_from_response(response)
            
            if error_msg:
                print(f"    - ⚠️ Could not parse refined plan: {error_msg}. Reverting.")
                return original_result
            
            # Safety check
            if "proposed_experiments" not in refined_result:
                print("    - ⚠️ Refined plan invalid structure. Reverting.")
                return original_result
                
            return refined_result
            
        except Exception as e:
            print(f"    - ⚠️ Error during refinement: {e}")
            return original_result
        

    def _display_plan_summary(self, result: Dict[str, Any]) -> None:
        """
        Parses the agent's results and prints a structured, pretty-printed 
        summary to the console for human review.
        """
        # 1. Error Handling
        if result.get("error"):
            print(f"\n❌ Agent finished with an error: {result['error']}\n")
            return

        # 2. structure Validation
        experiments = result.get("proposed_experiments")
        if not experiments or not isinstance(experiments, list):
            print("\n⚠️  The agent returned a result, but no experiments were found.")
            # Optional: Print raw if debugging needed
            # print(json.dumps(result, indent=2))
            return

        # 3. Header
        print("\n" + "="*80)
        print("✅ PROPOSED EXPERIMENTAL PLAN")
        print("="*80)

        # 4. Loop through Experiments
        for i, exp in enumerate(experiments, 1):
            
            # --- Name & Hypothesis ---
            print(f"\n🔬 EXPERIMENT {i}: {exp.get('experiment_name', 'Unnamed Experiment')}")
            print("-" * 80)
            print(f"\n> 🎯 Hypothesis:\n> {exp.get('hypothesis', 'N/A')}")

            # --- Experimental Steps (Numbered) ---
            print("\n--- 🧪 Experimental Steps ---")
            steps = exp.get('experimental_steps', [])
            if steps:
                for j, step in enumerate(steps, 1):
                    print(f" {j}. {step}")
            else:
                print("  (No steps provided)")
            
            # --- Equipment ---
            print("\n--- 🛠️  Required Equipment ---")
            equipment = exp.get('required_equipment', [])
            if equipment:
                # Print as a clean comma-separated list if short, or bullets if long
                if len(equipment) > 5:
                    for item in equipment: print(f"  * {item}")
                else:
                    print(f"  {', '.join(equipment)}")
            else:
                print("  (No equipment specified)")

            # --- Outcome & Justification (Critical for Review) ---
            print("\n--- 📈 Expected Outcome ---")
            print(f"  {exp.get('expected_outcome', 'N/A')}")

            print("\n--- 💡 Justification ---")
            print(f"  {exp.get('justification', 'N/A')}")
            
            # --- Source Documents ---
            print("\n--- 📄 Source Documents ---")
            sources = exp.get('source_documents', [])
            if sources:
                for src in sources:
                    print(f"  - {src}")
            else:
                print("  (No sources listed)")

            # --- Code Indicator (If generated) ---
            if "implementation_code" in exp:
                print("\n--- 💻 Implementation Code ---")
                print("  ✅ Python script generated (saved to file).")

            print("\n" + "="*80)