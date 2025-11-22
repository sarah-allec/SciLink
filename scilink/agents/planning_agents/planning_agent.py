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

    def _perform_rag_query(self, objective: str, instructions: str, task_name: str, 
                           primary_data_set: Optional[Dict[str, str]] = None,
                           image_paths: Optional[List[str]] = None,
                           image_descriptions: Optional[List[str]] = None,
                           additional_context: Optional[str] = None) -> Dict[str, Any]:
        
        # --- 1. Primary Data ---
        primary_data_str = None
        if primary_data_set:
            try:
                # Quick parse just for the summary
                chunks = parse_adaptive_excel(primary_data_set['file_path'], primary_data_set['metadata_path'])
                if chunks: 
                    # Try to find a summary chunk, else use the first one
                    summary = next((c for c in chunks if c['metadata'].get('content_type') in ('dataset_summary', 'dataset_package')), chunks[0])
                    primary_data_str = summary['text']
            except: pass

        # --- 2. Dual Retrieval ---
        print(f"\n--- Retrieving Context for {task_name} ---")
        
        # A. Retrieve Scientific Context (Top 10)
        doc_chunks = []
        if self.kb_docs.index and self.kb_docs.index.ntotal > 0:
            doc_chunks = self.kb_docs.retrieve(objective, top_k=10)
        
        # B. Retrieve Code Context (Top 5)
        code_chunks = []
        if self.kb_code.index and self.kb_code.index.ntotal > 0:
            code_query = f"{objective} implementation code API python"
            code_chunks = self.kb_code.retrieve(code_query, top_k=5)
            print(f"  - Retrieved {len(code_chunks)} code chunks from Code KB.")

        # Combine
        combined_chunks = doc_chunks + code_chunks
        unique_chunks = {c['text']: c for c in combined_chunks}.values()
        
        if not unique_chunks and not primary_data_str:
            retrieved_context_str = "No relevant context found."
        else:
            rag_str = "\n\n---\n\n".join(
                f"Source: {Path(c['metadata'].get('source', 'N/A')).name}\nType: {c['metadata'].get('content_type')}\n\n{c['text']}" 
                for c in unique_chunks
            )
            retrieved_context_str = ""
            if primary_data_str: retrieved_context_str += f"## Primary Data\n{primary_data_str}\n\n"
            if rag_str: retrieved_context_str += f"## Retrieved Context (Docs & Code)\n{rag_str}"

        # --- 3. Multimodal Prompt Construction ---
        loaded_images = []
        img_desc_str = ""
        if image_paths and PIL_Image:
            for p in image_paths:
                try: loaded_images.append(PIL_Image.open(p))
                except: pass
        if image_descriptions:
            img_desc_str = json.dumps(image_descriptions, indent=2)

        prompt_parts = [instructions, f"## Objective:\n{objective}"]
        if loaded_images:
            prompt_parts.append("\n## Provided Images: (See attached)")
            prompt_parts.extend(loaded_images)
            if img_desc_str: prompt_parts.append(f"\n## Image Descriptions:\n{img_desc_str}")
        if additional_context:
            prompt_parts.append(f"\n## Additional Context:\n{additional_context}")
        prompt_parts.append(f"\n## Retrieved Context:\n{retrieved_context_str}")

        print(f"\n--- Generating {task_name} ---")
        try:
            response = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
            result, error_msg = self._parse_json_from_response(response)
            if error_msg: return {"error": error_msg}

            # --- Fallback Check ---
            fallback_needed = False
            if result.get("error") and "Insufficient" in result.get("error"): fallback_needed = True
            elif instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS and not result.get("error"):
                 if not self._verify_plan_relevance(objective, result): fallback_needed = True

            if fallback_needed:
                print("    - ⚠️  Entering Fallback Mode...")
                # Use fallback instructions
                fb_inst = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK if instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS else TEA_INSTRUCTIONS_FALLBACK
                prompt_parts[0] = fb_inst
                fb_resp = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
                result, _ = self._parse_json_from_response(fb_resp)
                
                # If we have a Code KB, try to refine the fallback code
                if result and result.get("proposed_experiments") and self.kb_code.index and self.kb_code.index.ntotal > 0:
                     print("    - 🔍 Attempting code refinement on fallback plan...")
                     plan_txt = " ".join([e.get('experiment_name') for e in result['proposed_experiments']])
                     hits = self.kb_code.retrieve(f"implementation for {plan_txt}", top_k=5)
                     if hits:
                        print(f"    - ✅ Found {len(hits)} specific code chunks. Refining...")
                        code_ctx = "\n\n".join([c['text'] for c in hits])

                        # 1. FIX: Define the variable before using it
                        found_code_files = list(set([Path(c['metadata']['source']).name for c in hits]))

                        # 2. FIX: Use the "Syntax Guide" prompt to prevent mismatch errors
                        refine_prompt = f"""
                        You are an expert Research Software Engineer.
                        
                        TASK: 
                        Write an executable Python script to automate the "Experimental Steps" provided below.
                        
                        INPUTS:
                        1. Experimental Logic: {json.dumps([e.get('experimental_steps') for e in result['proposed_experiments']])}
                        2. Syntax Reference (Context): {code_ctx}
                        
                        INSTRUCTIONS:
                        - The "Syntax Reference" contains valid API commands and coding patterns.
                        - The "Experimental Logic" contains the specific scientific actions.
                        - **CRITICAL:** Use the *coding patterns* from the Syntax Reference to implement the *actions* from the Experimental Logic. 
                        - Do NOT expect the Syntax Reference to contain specific chemical names. It is a manual, not a recipe.
                        - Remove any "Warning" strings about general knowledge; this code is now grounded.
                        
                        OUTPUT: 
                        Respond with a SINGLE JSON object: {{ "final_implementation_code": "THE_CODE_HERE" }}
                        """
                        r_resp = self.model.generate_content([refine_prompt], generation_config=self.generation_config)
                        r_json, _ = self._parse_json_from_response(r_resp)
                        if r_json and r_json.get("final_implementation_code"):
                            for exp in result["proposed_experiments"]:
                                exp["implementation_code"] = r_json["final_implementation_code"]
                                exp["code_source_files"] = found_code_files
                                exp["source_documents"].extend([Path(c['metadata']['source']).name for c in hits])
                            print("    - ✅ Code refined.")
                        else:
                            print(f"    - ❌ Refinement failed. Error: {error}")

            return result
        except Exception as e:
            return {"error": str(e)}

    def propose_experiments(self, objective: str, 
                            science_paths: Optional[List[str]] = None, 
                            code_paths: Optional[List[str]] = None,
                            structured_data_sets: Optional[List[Dict[str, str]]] = None,
                            tea_summary: Optional[str] = None,
                            primary_data_set: Optional[Dict[str, str]] = None,
                            image_paths: Optional[List[str]] = None,
                            image_descriptions: Optional[List[str]] = None,
                            output_json_path: Optional[str] = None):
        """Generates experimental plans using Dual-KB retrieval."""
        
        if not self._ensure_kb_is_ready(science_paths, code_paths, structured_data_sets):
            return {"error": "KB Init Failed"}

        ctx = f"TEA Findings:\n{tea_summary}" if tea_summary else None
        res = self._perform_rag_query(objective, HYPOTHESIS_GENERATION_INSTRUCTIONS, "Experimental Plan", 
                                      primary_data_set, image_paths, image_descriptions, ctx)
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

        res = self._perform_rag_query(objective, TEA_INSTRUCTIONS, "Technoeconomic Analysis",
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