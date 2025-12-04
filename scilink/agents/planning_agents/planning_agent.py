import google.generativeai as genai
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass, chunk_text
from .excel_parser import parse_adaptive_excel
from .parser_utils import get_files_from_directory, generate_repo_map
from .repo_loader import clone_git_repository

from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS
)

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel

from .rag_engine import (
    perform_science_rag, 
    perform_code_rag, 
    refine_plan_with_feedback,
    verify_plan_relevance
)
from .user_interface import display_plan_summary, get_user_feedback


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
                 code_chunk_size: int = 20000): 
        
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
        self.kb_code_map_path = str(self.kb_code_prefix.with_suffix(".maps.json"))

        print("--- Initializing Agent (Dual-KB System) ---")
        self._load_knowledge_bases()

    def _load_knowledge_bases(self):
        """Attempts to load both KBs from disk."""
        print(f"  - Docs KB: Loading from {self.kb_docs_prefix}...")
        docs_loaded = self.kb_docs.load(self.kb_docs_index, self.kb_docs_chunks)
        
        print(f"  - Code KB: Loading from {self.kb_code_prefix}...")
        code_loaded = self.kb_code.load(self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path)

        self._kb_is_built = docs_loaded or code_loaded
        
        if docs_loaded: print("    - ✅ Docs KB loaded.")
        if code_loaded: print("    - ✅ Code KB loaded.")
        if not self._kb_is_built: print("    - ⚠️  No pre-built KBs found.")

    def _save_results_to_json(self, results: Dict[str, Any], file_path: str):
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            print(f"    - ✅ Results successfully saved to: {file_path}")
        except Exception as e: logging.error(f"    - ❌ Failed to save results: {e}")

    def _process_file_list(self, file_paths: List[str], is_code_mode: bool, repo_name: str = None) -> List[Dict[str, Any]]:
        """
        Generic helper to process a list of files OR directories.
        If is_code_mode=True, treats text files as code blocks and tags metadata as 'code'.
        """
        chunks = []

        expanded_paths = []
        if file_paths:
            for f_path in file_paths:
                path_obj = Path(f_path)
                if path_obj.is_dir():
                    # If user provided a folder (e.g., "./neurobayes"), get all files inside
                    expanded_paths.extend(get_files_from_directory(f_path))
                else:
                    # If user provided a specific file (e.g., "neurobayes_dump.txt"), keep it
                    expanded_paths.append(f_path)

        for f_path in expanded_paths:
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
                    # See every file processed
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
            
            for p in code_paths:
                path_obj = Path(p)
                
                # Case A: Directory (Repo) - Generate Map & Tag Chunks
                if path_obj.is_dir():
                    repo_name = path_obj.name
                    print(f"  - 📦 Processing Repo: {repo_name}")
                    
                    # Store map in KB registry
                    self.kb_code.repo_maps[repo_name] = generate_repo_map(str(path_obj))
                    
                    # Process chunks with repo_name tag
                    # Note: We pass [p] so helper expands this specific folder
                    repo_chunks = self._process_file_list([p], is_code_mode=True, repo_name=repo_name)
                    code_chunks.extend(repo_chunks)
                
                # Case B: Individual File
                else:
                    file_chunks = self._process_file_list([p], is_code_mode=True)
                    code_chunks.extend(file_chunks)
            
        if code_chunks:
            print(f"  - Building Code KB with {len(code_chunks)} chunks...")
            self.kb_code.build(code_chunks)
            self.kb_code.save(
                self.kb_code_index, 
                self.kb_code_chunks, 
                self.kb_code_map_path
            )
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

    def propose_experiments(self, objective: str, 
                            science_paths: Optional[List[str]] = None, 
                            code_paths: Optional[List[str]] = None,
                            structured_data_sets: Optional[List[Dict[str, str]]] = None,
                            additional_context: Optional[Dict[str, str]] = None,
                            primary_data_set: Optional[Dict[str, str]] = None,
                            image_paths: Optional[List[str]] = None,
                            image_descriptions: Optional[List[str]] = None,
                            output_json_path: Optional[str] = None,
                            enable_human_feedback: bool = True):
        """
        Orchestrates the full end-to-end generation of experimental plans, from scientific 
        theory to executable code implementation.

        This method employs a Phased RAG Architecture:
        1. **Phase 1 (Science):** Uses the `science_paths` KB to generate hypotheses and logical steps.
           Includes autonomous self-reflection to verify plan relevance.
        2. **Phase 2 (Feedback):** (Optional) Pauses for human review via CLI to refine the strategy.
        3. **Phase 3 (Engineering):** Uses the `code_paths` KB to map approved steps to Python code.

        Args:
            objective (str): The high-level research goal or question (e.g., "Synthesize X using Y").
            science_paths (List[str], optional): List of local file paths (PDFs, TXT, MD) containing 
                scientific literature, manuals, or reports.
            code_paths (List[str], optional): List of sources for implementation context. 
                **Supports Polymorphic Inputs:**
                - Local file paths (e.g., `"./scripts/utils.py"`)
                - Local directory paths (e.g., `"./legacy_codebase/"`)
                - **Git URLs** (e.g., `"https://github.com/org/repo.git"`). Remote repos are 
                  automatically cloned/updated and ingested.
            structured_data_sets (List[Dict], optional): List of Excel/CSV metadata for general context. 
                Format: `[{'file_path': '...', 'metadata_path': '...'}]`.
            additional_context (Dict[str, str], optional): A dictionary of extra text context.
                The keys will be used as headers and values as content in the prompt.
                Example: {
                    "TEA Findings": "Platinum is too expensive.", 
                    "Safety Constraints": "Do not use HF."
                }
            primary_data_set (Dict, optional): A specific dataset that is the focus of this experiment.
            image_paths (List[str], optional): Paths to relevant images (charts, diagrams) for multimodal analysis.
            image_descriptions (List[str], optional): Contextual text descriptions for the provided images.
            output_json_path (str, optional): If provided, saves the final result dictionary to this JSON file.
            enable_human_feedback (bool): If True, pauses execution after Phase 1 to allow the user to 
                critique or approve the plan via the console. Defaults to True.

        Returns:
            Dict[str, Any]: A structured dictionary containing:
                - "proposed_experiments": List of experimental plans.
                - "implementation_code": Python scripts for the experiments (if Phase 3 succeeds).
                - "error": Error message if the pipeline failed.
        """

        effective_code_paths = []
        
        if code_paths:
            print("\n--- Resolving Code Paths ---")
            for path in code_paths:
                # Check if the string looks like a Git URL
                if path.strip().startswith(('http://', 'https://', 'git@')):
                    print(f"  - 🔗 Detected URL: {path}")
                    local_path = clone_git_repository(path)
                    if local_path:
                        effective_code_paths.append(local_path)
                        print(f"    -> Resolved to local: {Path(local_path).name}")
                else:
                    # It's a normal local file/folder
                    effective_code_paths.append(path)

        # --- Init Knowledge Bases ---
        if not self._ensure_kb_is_ready(science_paths, effective_code_paths, structured_data_sets):
            return {"error": "KB Init Failed"}

        # =====================================================
        # PHASE 1: SCIENCE STRATEGY (Docs KB Only)
        # =====================================================
        print(f"\n--- Phase 1: Generating Experimental Strategy ---")
        
        # Iterate through the dictionary to create a structured string
        ctx_string = ""
        if additional_context:
            for header, content in additional_context.items():
                ctx_string += f"## {header}\n{content}\n\n"
        
        ctx_string = ctx_string.strip() if ctx_string else None
        
        # This generates the Plan, Steps, and Hypothesis (No Code)
        res = perform_science_rag(
            objective=objective,
            instructions=HYPOTHESIS_GENERATION_INSTRUCTIONS,
            task_name="Experimental Plan",
            kb_docs=self.kb_docs,             
            model=self.model,                 
            generation_config=self.generation_config,
            primary_data_set=primary_data_set,
            image_paths=image_paths,
            image_descriptions=image_descriptions,
            additional_context=ctx_string # Pass the formatted string
        )

        # Self-reflection and correction
        if not res.get("error"):
            # Check relevance
            is_relevant, critique = verify_plan_relevance(objective, res, self.model, self.generation_config)
            
            if not is_relevant:
                print(f"\n🔄 Self-Reflection triggered: {critique}")
                print("    - Attempting autonomous plan correction...")
   
                res = refine_plan_with_feedback(
                    original_result=res,
                    feedback=f"CRITICAL CORRECTION NEEDED: {critique}. Ensure the plan directly addresses the objective: {objective}",
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                print("    - ✅ Plan auto-corrected.")
                
        # =====================================================
        # PHASE 2: HUMAN FEEDBACK LOOP
        # =====================================================
        if enable_human_feedback and res.get("proposed_experiments") and not res.get("error"):
            
            # Show the Science Plan
            display_plan_summary(res)
            # Capture user input
            user_feedback = get_user_feedback()
            
            if user_feedback:
                print(f"\n📝 Feedback received. Refining Scientific Plan...")
                res = refine_plan_with_feedback(
                    original_result=res,
                    feedback=user_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                display_plan_summary(res)
                print("✅ Scientific plan updated.")
            else:
                print("✅ Scientific plan accepted.")

        # =====================================================
        # PHASE 3: CODE IMPLEMENTATION (Code KB Only)
        # =====================================================
        if self.kb_code.index and self.kb_code.index.ntotal > 0 and not res.get("error"):
             print(f"\n--- Phase 3: Mapping to Implementation Code ---")
             res = perform_code_rag(
                 result=res,
                 kb_code=self.kb_code,
                 model=self.model,
                 generation_config=self.generation_config
             )

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

        res = perform_science_rag(
            objective=objective, 
            instructions=TEA_INSTRUCTIONS, 
            task_name="Technoeconomic Analysis",
            kb_docs=self.kb_docs,
            model=self.model,
            generation_config=self.generation_config,
            primary_data_set=primary_data_set, 
            image_paths=image_paths, 
            image_descriptions=image_descriptions
        )

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