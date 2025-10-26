import google.generativeai as genai
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass
from .csv_parser import parse_csv_with_context
from .excel_parser import parse_adaptive_excel
from .instruct import HYPOTHESIS_GENERATION_INSTRUCTIONS, TEA_INSTRUCTIONS
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

    def _build_and_save_kb(self, pdf_paths: List[str] = None, experimental_data: List[Dict[str, str]] = None) -> bool:
        """Internal method to parse documents, build the KB, and save it."""
        pdf_paths = pdf_paths or []
        experimental_data = experimental_data or []
        print("\n--- Parsing all provided documents ---")
        all_chunks = []
        for pdf_path in pdf_paths:
             all_chunks.extend(extract_pdf_two_pass(pdf_path))
        for data_pair in experimental_data:
            data_path = data_pair.get('data_path')
            context_path = data_pair.get('context_path')
            if not data_path or not context_path:
                print(f"  - ⚠️  Skipping data pair: Missing 'data_path' or 'context_path'.")
                continue
            file_ext = Path(data_path).suffix.lower()
            if file_ext == '.csv':
                csv_chunk = parse_csv_with_context(data_path, context_path)
                if csv_chunk: all_chunks.append(csv_chunk)
            elif file_ext in ['.xlsx', '.xls']:
                excel_chunks = parse_adaptive_excel(data_path, context_path)
                if excel_chunks: all_chunks.extend(excel_chunks)
            else:
                print(f"  - ⚠️  Skipping unsupported file type: {data_path}")
        if not all_chunks:
            print("❌ Build failed: No content extracted.")
            self._kb_is_built = False
            return False
        print("\n--- Building Knowledge Base (Embedding & Indexing) ---")
        self.knowledge_base.build(all_chunks)
        print("\n--- Saving Knowledge Base to Disk ---")
        self.knowledge_base.save(self.index_file, self.chunks_file)
        self._kb_is_built = True
        print(f"✅ KB built with {len(all_chunks)} chunks, saved to {self.kb_path_prefix}, ready.")
        return True

    def _ensure_kb_is_ready(self, pdf_paths: Optional[List[str]] = None, experimental_data: Optional[List[Dict[str, str]]] = None) -> bool:
        """Checks if KB is built or builds it if new documents are provided."""
        new_documents_provided = (pdf_paths is not None and pdf_paths) or \
                                 (experimental_data is not None and experimental_data)
        if new_documents_provided:
            print("--- New documents provided. Building/rebuilding knowledge base... ---")
            build_success = self._build_and_save_kb(pdf_paths, experimental_data)
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
                           additional_context: Optional[str] = None) -> Dict[str, Any]:
        """Internal helper for retrieval, prompt construction, and LLM generation."""
        print(f"\n--- Retrieving Relevant Context for {task_name} ---")
        context_chunks = self.knowledge_base.retrieve(objective, top_k=7) # shall we pass top_k as an argument?

        if not context_chunks:
            logging.warning(f"Could not retrieve relevant context for {task_name} objective: '{objective}'")
            retrieved_context_str = "No relevant context found in the provided documents."
        else:
            retrieved_context_str = "\n\n---\n\n".join(
                f"Source: {Path(chunk['metadata'].get('source', 'N/A')).name}\n\n{chunk['text']}"
                for chunk in context_chunks
            )

        # Construct Prompt
        prompt_parts = [instructions]
        prompt_parts.append(f"## Objective:\n{objective}")

        # Add optional additional context (like TEA summary) if provided
        if additional_context:
            prompt_parts.append(f"## Additional Context/Findings:\n{additional_context}")
            prompt_parts.append("\n**IMPORTANT:** Consider this additional context when generating your response.**")

        prompt_parts.append(f"## Retrieved Context from Documents:\n{retrieved_context_str}")

        prompt = "\n\n".join(prompt_parts)

        print(f"\n--- Generating {task_name} with LLM ---")
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)

            # Parsing logic
            if hasattr(response, 'text'):
                json_text = response.text.strip()
                if json_text.startswith("```json"): json_text = json_text[len("```json"):].strip()
                if json_text.endswith("```"): json_text = json_text[:-len("```")].strip()
                result = json.loads(json_text)
                print(f"  - ✅ Successfully generated and parsed {task_name}.")
                return result
            elif hasattr(response, 'parts') and response.parts:
                 json_text = response.parts[0].text.strip()
                 if json_text.startswith("```json"): json_text = json_text[len("```json"):].strip()
                 if json_text.endswith("```"): json_text = json_text[:-len("```")].strip()
                 result = json.loads(json_text)
                 print(f"  - ✅ Successfully generated and parsed {task_name} from response part.")
                 return result
            else:
                 logging.error(f"LLM response format unexpected: {response}")
                 return {"error": "LLM response format unexpected.", "raw_response": str(response)}

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
                            pdf_paths: Optional[List[str]] = None,
                            experimental_data: Optional[List[Dict[str, str]]] = None,
                            tea_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates experimental proposals. Builds/rebuilds KB if new documents provided.
        Optionally incorporates findings from a previous TEA via tea_summary.
        """
        if not self._ensure_kb_is_ready(pdf_paths, experimental_data):
            return {"error": "Knowledge base preparation failed."}

        # Format the TEA summary for the prompt helper
        additional_context_for_prompt = None
        if tea_summary:
            additional_context_for_prompt = f"Key Findings from Prior Technoeconomic Analysis:\n{tea_summary}"

        # Call the helper with specific instructions and optional TEA context
        return self._perform_rag_query(
            objective=objective,
            instructions=HYPOTHESIS_GENERATION_INSTRUCTIONS,
            task_name="Experimental Plan",
            additional_context=additional_context_for_prompt # Pass formatted summary
        )

    def perform_technoeconomic_analysis(self,
                                        objective: str,
                                        pdf_paths: Optional[List[str]] = None,
                                        experimental_data: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Performs preliminary TEA. Builds/rebuilds KB if new documents provided.
        """
        if not self._ensure_kb_is_ready(pdf_paths, experimental_data):
            return {"error": "Knowledge base preparation failed."}

        # Call the helper with specific instructions, no additional context needed here
        return self._perform_rag_query(
            objective=objective,
            instructions=TEA_INSTRUCTIONS,
            task_name="Technoeconomic Analysis",
            additional_context=None # TEA doesn't take prior context
        )