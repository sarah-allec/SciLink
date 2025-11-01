import google.generativeai as genai
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import PIL.Image
PIL_Image = PIL.Image


from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass
from .excel_parser import parse_adaptive_excel
from .instruct import HYPOTHESIS_GENERATION_INSTRUCTIONS, TEA_INSTRUCTIONS, HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel

class PlanningAgent:
    """
    Orchestrates RAG pipelines to generate experimental hypotheses
    and perform preliminary technoeconomic analysis from documents.
    """
    def __init__(self, google_api_key: str = None,
                 model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 embedding_model: str = "gemini-embedding-001",
                 kb_base_path: str = "./kb_storage/default_kb"):
        """
        Initializes the agent, LLM backends, and loads the knowledge base
        from the specified disk location if it exists.
        """

        # Need to replace it with scilink.auth
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')

        # LLM Backend Configuration
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  Using OpenAI-compatible model for generation: {model_name}")
            self.model = OpenAIAsGenerativeModel(model_name, api_key=google_api_key, base_url=local_model)
            self.generation_config = None
        else:
            logging.info(f"☁️  Using Google Gemini model for generation: {model_name}")
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        # KnowledgeBase Initialization
        self.knowledge_base = KnowledgeBase(
            google_api_key=google_api_key,
            embedding_model=embedding_model,
            local_model=local_model
        )

        # KnowledgeBase Persistence Paths
        self.kb_path_prefix = Path(kb_base_path)
        self.kb_path_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.index_file = str(self.kb_path_prefix.with_suffix(".faiss"))
        self.chunks_file = str(self.kb_path_prefix.with_suffix(".json"))

        print("--- Initializing Agent ---")
        print(f"Attempting to load Knowledge Base from: {self.kb_path_prefix}.(faiss/json)")
        self._kb_is_built = self.knowledge_base.load(self.index_file, self.chunks_file)
        if self._kb_is_built:
            print("✅ Knowledge base loaded successfully.")
        else:
            print("⚠️  No pre-built KB found. Provide documents on your first call to create one.")

    def _save_results_to_json(self, results: Dict[str, Any], file_path: str):
        """Helper to save dictionary results to a JSON file."""
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"    - ✅ Results successfully saved to: {file_path}")
        except Exception as e:
            logging.error(f"    - ❌ Failed to save results to {file_path}: {e}")

    def _build_and_save_kb(self,
                           document_paths: Optional[List[str]] = None,
                           structured_data_sets: Optional[List[Dict[str, str]]] = None
                           ) -> bool:
        """
        Internal method to parse documents, build the KB, and save it.
        Uses document_paths for general documents (like PDFs) and
        structured_data_sets for tabular data with context files.
        Returns True on success, False on failure.
        """
        doc_paths = document_paths or []
        struct_data = structured_data_sets or []

        print("\n--- Parsing all provided documents ---")
        all_chunks = []

        # --- 1. Process General Documents (PDFs) ---
        for doc_path in doc_paths:
            if Path(doc_path).suffix.lower() == '.pdf':
                all_chunks.extend(extract_pdf_two_pass(doc_path))
            else:
                print(f"  - ⚠️  Skipping unsupported file type in document_paths: {doc_path}")

        # --- 2. Process Structured Data Sets (e.g., Excel + JSON) ---
        for data_set in struct_data:
            file_path = data_set.get('file_path')
            metadata_path = data_set.get('metadata_path')

            if not file_path or not metadata_path:
                print(f"  - ⚠️  Skipping data set: Missing 'file_path' or 'metadata_path' in dictionary: {data_set}")
                continue

            file_ext = Path(file_path).suffix.lower()

            try:
                if file_ext in ['.xlsx', '.xls']:
                    excel_chunks = parse_adaptive_excel(file_path, metadata_path)
                    if excel_chunks:
                        all_chunks.extend(excel_chunks)
                else:
                    print(f"  - ⚠️  Skipping unsupported file type in structured_data_sets: {file_path}")
            except Exception as e:
                print(f"  - ❌ Error processing data pair: file='{file_path}', metadata='{metadata_path}'. Error: {e}")
                continue # Continue with the next file pair

        # --- 3. Build and Save KB ---
        if not all_chunks:
            print("❌ Build failed: No content successfully extracted from any provided documents.")
            self._kb_is_built = False
            return False

        print(f"\n--- Building Knowledge Base (Embedding & Indexing) from {len(all_chunks)} chunks ---")
        try:
            self.knowledge_base.build(all_chunks)
        except Exception as e:
            print(f"❌ Build failed: Error during embedding or indexing: {e}")
            self._kb_is_built = False
            return False

        print("\n--- Saving Knowledge Base to Disk ---")
        try:
            self.knowledge_base.save(self.index_file, self.chunks_file)
            self._kb_is_built = True
            print(f"✅ KB built with {len(all_chunks)} chunks, saved to {self.kb_path_prefix}, ready.")
            return True
        except Exception as e:
             print(f"❌ Build successful, but failed to save KB: {e}")
             self._kb_is_built = True
             return True

    def _ensure_kb_is_ready(self, document_paths: Optional[List[str]] = None, structured_data_sets: Optional[List[Dict[str, str]]] = None) -> bool:
        """Checks if KB is built or builds it if new documents are provided."""
        new_documents_provided = (document_paths is not None and document_paths) or \
                                 (structured_data_sets is not None and structured_data_sets)
        if new_documents_provided:
            print("--- New documents provided. Building/rebuilding knowledge base... ---")
            build_success = self._build_and_save_kb(document_paths, structured_data_sets)
            if not build_success:
                logging.error("Failed to build knowledge base from provided documents.")
                return False
        elif not self._kb_is_built:
            logging.error("Knowledge base is not built. Please provide documents.")
            return False
        else:
             print("--- Using existing knowledge base. ---")
        return True

    def _perform_rag_query(self,
                           objective: str,
                           instructions: str,
                           task_name: str,
                           additional_context: Optional[str] = None,
                           primary_data_set: Optional[Dict[str, str]] = None,
                           image_paths: Optional[List[str]] = None,
                           image_descriptions: Optional[List[str | Dict[str, Any]]] = None
                           ) -> Dict[str, Any]:
        """Internal helper for retrieval, prompt construction, and LLM generation."""
        
        # --- 1. Process the Primary Data Set ---
        primary_data_context_str = None
        if primary_data_set:
            print(f"  - ℹ️  Parsing primary data set to force into context...")
            try:
                file_path = primary_data_set['file_path']
                meta_path = primary_data_set['metadata_path']
                # Parse the file (this returns a list of chunks)
                data_chunks = parse_adaptive_excel(file_path, meta_path)
                
                if data_chunks:
                    # Find the summary/package chunk
                    summary_chunk = next(
                        (c for c in data_chunks if c['metadata'].get('content_type') in ('dataset_summary', 'dataset_package')),
                        data_chunks[0] # Fallback to first chunk if no summary found
                    )
                    primary_data_context_str = summary_chunk['text']
                    print(f"  - ✅ Successfully parsed and loaded primary data from '{Path(file_path).name}'.")
                else:
                    print(f"  - ⚠️  Primary data set '{Path(file_path).name}' was provided but parsing returned no chunks.")
            except Exception as e:
                print(f"  - ❌ Error parsing primary data set: {e}")

        # --- 2. Perform RAG Retrieval from the Knowledge Base ---
        print(f"\n--- Retrieving Relevant Context for {task_name} ---")
        context_chunks = self.knowledge_base.retrieve(objective, top_k=7) 

        # We use a dict with text as the key to auto-deduplicate
        final_chunks = {chunk['text']: chunk for chunk in context_chunks}
        
        # --- 3. Build the Final Context String ---
        if not final_chunks and not primary_data_context_str:
            logging.warning(f"Could not retrieve any relevant context for {task_name} objective: '{objective}'")
            retrieved_context_str = "No relevant context found in the provided documents."
        else:
            # Build the context string from RAG retrieval
            rag_context_str = "\n\n---\n\n".join(
                f"Source: {Path(chunk['metadata'].get('source', 'N/A')).name}\n\n{chunk['text']}"
                for chunk in final_chunks.values()
            )
            
            # Combine primary data and RAG context
            full_context_parts = []
            if primary_data_context_str:
                full_context_parts.append(f"## Primary Data Context (From User-Provided File)\n{primary_data_context_str}")
            if rag_context_str:
                full_context_parts.append(f"## Retrieved Context (From Knowledge Base)\n{rag_context_str}")
            
            retrieved_context_str = "\n\n".join(full_context_parts)

        # --- 4. Load Images and Process Descriptions ---
        loaded_images = []
        image_descriptions_str_parts = []
        
        # 4.1 Load Images from paths
        if image_paths and PIL_Image:
            print(f"  - ℹ️  Loading {len(image_paths)} image(s)...")
            for img_path in image_paths:
                try:
                    image = PIL_Image.open(img_path)
                    loaded_images.append(image)
                    print(f"    - ✅ Loaded image: {img_path}")
                except Exception as e:
                    print(f"    - ❌ Failed to load image {img_path}: {e}")
        elif image_paths:
            print("  - ⚠️  Image paths provided, but PIL (Pillow) library is not installed. Images will be ignored.")

        # 4.2 Process Image Descriptions
        if image_descriptions:
            print(f"  - ℹ️  Processing {len(image_descriptions)} image description(s)...")
            for i, desc in enumerate(image_descriptions):
                if isinstance(desc, dict):
                    # Assume it's a JSON/dict object, pretty-print it
                    try:
                        desc_str = json.dumps(desc, indent=2)
                        image_descriptions_str_parts.append(f"Description for Image {i+1} (JSON):\n{desc_str}")
                    except Exception as e:
                        print(f"    - ❌ Failed to serialize description dict: {e}")
                        image_descriptions_str_parts.append(f"Description for Image {i+1} (unserializable dict):\n{str(desc)}")
                elif isinstance(desc, str):
                    # It's a plain text string
                    image_descriptions_str_parts.append(f"Description for Image {i+1} (Text):\n{desc}")
                else:
                    print(f"    - ⚠️  Skipping unknown description format: {type(desc)}")
        
        image_descriptions_context_str = "\n\n".join(image_descriptions_str_parts)

        # --- 5. Construct Prompt (as a list for multimodal input) ---
        prompt_parts = []
        prompt_parts.append(instructions)
        prompt_parts.append(f"## Objective:\n{objective}")

        # Add images after the objective
        if loaded_images:
            prompt_parts.append("\n## Provided Images:\n(See attached images for visual context)\n")
            prompt_parts.extend(loaded_images)
            
            if image_descriptions_context_str:
                prompt_parts.append(f"\n## Provided Image Descriptions:\n(See attached descriptions for additional context)\n{image_descriptions_context_str}\n")
        
        if additional_context:
            prompt_parts.append(f"\n## Additional Context/Findings:\n{additional_context}")
            prompt_parts.append("\n**IMPORTANT:** Consider this additional context when generating your response.**")

        prompt_parts.append(f"\n## Retrieved Context from Documents:\n{retrieved_context_str}")

        print(f"\n--- Generating {task_name} with LLM ---")
        try:
            # --- First Attempt ---
            response = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
            
            # Helper function to parse JSON from response text
            def parse_json_from_response(resp):
                if hasattr(resp, 'text'):
                    json_text = resp.text.strip()
                elif hasattr(resp, 'parts') and resp.parts:
                    json_text = resp.parts[0].text.strip()
                else:
                    return None, f"LLM response format unexpected: {resp}"
                
                if json_text.startswith("```json"): json_text = json_text[len("```json"):].strip()
                if json_text.endswith("```"): json_text = json_text[:-len("```")].strip()
                
                return json.loads(json_text), None

            result, error_msg = parse_json_from_response(response)
            if error_msg:
                logging.error(error_msg)
                return {"error": error_msg, "raw_response": str(response)}

            # --- FALLBACK LOGIC ---
            if (result.get("error") and 
                "Insufficient context" in result.get("error") and 
                instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS):
                
                print(f"    - ⚠️  Insufficient context. Attempting fallback with general knowledge...")
                
                fallback_prompt_parts = [HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK]
                fallback_prompt_parts.append(f"## Objective:\n{objective}")

                # Add images to fallback prompt as well
                if loaded_images:
                    fallback_prompt_parts.append("\n## Provided Images:\n(See attached images for visual context)\n")
                    fallback_prompt_parts.extend(loaded_images)

                    if image_descriptions_context_str:
                        fallback_prompt_parts.append(f"\n## Provided Image Descriptions:\n(See attached descriptions for additional context)\n{image_descriptions_context_str}\n")
                
                if additional_context:
                    fallback_prompt_parts.append(f"\n## Additional Context/Findings:\n{additional_context}")
                    fallback_prompt_parts.append("\n**IMPORTANT:** Consider this additional context when generating your response.**")

                fallback_prompt_parts.append(f"\n## Retrieved Context from Documents:\n{retrieved_context_str}")
                
                fallback_response = self.model.generate_content(fallback_prompt_parts, generation_config=self.generation_config)
                
                result, error_msg = parse_json_from_response(fallback_response)
                if error_msg:
                    logging.error(f"Fallback attempt failed to parse: {error_msg}")
                    return {"error": f"Fallback attempt failed: {error_msg}", "raw_response": str(fallback_response)}
                
                print(f"    - ✅ Successfully generated fallback {task_name}.")
                return result

            print(f"    - ✅ Successfully generated and parsed {task_name}.")
            return result

        except json.JSONDecodeError as json_e:
            raw_response_text = "N/A"
            if 'response' in locals() and hasattr(response, 'text'): raw_response_text = response.text
            elif 'response' in locals() and hasattr(response, 'parts') and response.parts: raw_response_text = response.parts[0].text if response.parts[0].text else str(response.parts)
            logging.error(f"Failed to parse LLM JSON response: {json_e}")
            return {"error": f"Failed to parse LLM JSON response: {str(json_e)}", "raw_response": raw_response_text}
        except Exception as e:
            raw_response_text = "N/A"
            if 'response' in locals() and hasattr(response, 'text'): raw_response_text = response.text
            elif 'response' in locals() and hasattr(response, 'parts') and response.parts: raw_response_text = response.parts[0].text if response.parts[0].text else str(response.parts)
            logging.error(f"Failed to generate LLM response: {e}")
            return {"error": f"Failed to generate LLM response: {str(e)}", "raw_response": raw_response_text}

    def propose_experiments(self,
                            objective: str,
                            document_paths: Optional[List[str]] = None,
                            structured_data_sets: Optional[List[Dict[str, str]]] = None,
                            tea_summary: Optional[str] = None,
                            primary_data_set: Optional[Dict[str, str]] = None,
                            image_paths: Optional[List[str]] = None,
                            image_descriptions: Optional[List[str | Dict[str, Any]]] = None,
                            output_json_path: Optional[str] = None
                            ) -> Dict[str, Any]:
        """
        Generates experimental proposals. Builds/rebuilds KB if new documents provided.
        Optionally incorporates findings from a previous TEA via tea_summary.
        
        Args:
            objective: The main objective for the proposals (used for RAG retrieval).
            document_paths: List of PDFs (to build the Knowledge Base).
            structured_data_sets: List of Excel/JSON pairs (to build the Knowledge Base).
            tea_summary: (Optional) A summary from a previous TEA to add as context.
            primary_data_set: (Recommended) A single Excel/JSON pair to be
                force-fed into the prompt as the primary context.
            image_paths: (Optional) A list of file paths to images for visual context.
            image_descriptions: (Optional) A list of text strings or JSON/dicts
                describing images.
            output_json_path: (Optional) A file path to save the JSON results.
        """
        if not self._ensure_kb_is_ready(document_paths, structured_data_sets):
            return {"error": "Knowledge base preparation failed."}

        # Format the TEA summary for the prompt helper
        additional_context_for_prompt = None
        if tea_summary:
            additional_context_for_prompt = f"Key Findings from Prior Technoeconomic Analysis:\n{tea_summary}"

        # Call the helper with specific instructions and optional TEA context
        results = self._perform_rag_query(
            objective=objective,
            instructions=HYPOTHESIS_GENERATION_INSTRUCTIONS,
            task_name="Experimental Plan",
            additional_context=additional_context_for_prompt,
            primary_data_set=primary_data_set,
            image_paths=image_paths,
            image_descriptions=image_descriptions
        )

        if output_json_path:
            print(f"\n--- Saving Experimental Plan Results ---")
            self._save_results_to_json(results, output_json_path)

        return results

    def perform_technoeconomic_analysis(self,
                                        objective: str,
                                        document_paths: Optional[List[str]] = None,
                                        structured_data_sets: Optional[List[Dict[str, str]]] = None,
                                        primary_data_set: Optional[Dict[str, str]] = None,
                                        image_paths: Optional[List[str]] = None,
                                        image_descriptions: Optional[List[str | Dict[str, Any]]] = None,
                                        output_json_path: Optional[str] = None
                                        ) -> Dict[str, Any]:
        """
        Performs preliminary TEA. Builds/rebuilds KB if new documents provided.
        
        Args:
            objective: The main objective for the TEA (used for RAG retrieval).
            document_paths: List of PDFs (to build the Knowledge Base).
            structured_data_sets: List of Excel/JSON pairs (to build the Knowledge Base).
            primary_data_set: (Recommended) A single Excel/JSON pair to be
                force-fed into the prompt as the primary context.
            image_paths: (Optional) A list of file paths to images for visual context.
            image_descriptions: (Optional) A list of text strings or JSON/dicts
                describing images.
            output_json_path: (Optional) A file path to save the JSON results.
        """
        if not self._ensure_kb_is_ready(document_paths, structured_data_sets):
            return {"error": "Knowledge base preparation failed."}

        # Call the helper with specific instructions, no additional context needed here
        results = self._perform_rag_query(
            objective=objective,
            instructions=TEA_INSTRUCTIONS,
            task_name="Technoeconomic Analysis",
            additional_context=None,
            primary_data_set=primary_data_set,
            image_paths=image_paths,
            image_descriptions=image_descriptions
        )

        if output_json_path:
            print(f"\n--- Saving TEA Results ---")
            self._save_results_to_json(results, output_json_path)

        return results