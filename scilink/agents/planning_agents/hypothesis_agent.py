import google.generativeai as genai
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass
from .csv_parser import parse_csv_with_context
from .instruct import HYPOTHESIS_GENERATION_INSTRUCTIONS
from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel

class HypothesisGeneratorAgent:
    """
    Orchestrates the RAG pipeline to generate experimental hypotheses from documents.
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
        
        # Attempt to load KB on initialization
        print("--- Initializing Agent ---")
        print(f"Attempting to load Knowledge Base from: {self.kb_path_prefix}.(faiss/json)")
        self._kb_is_built = self.knowledge_base.load(self.index_file, self.chunks_file)
        
        if self._kb_is_built:
            print("✅ Knowledge base loaded successfully.")
        else:
            print("⚠️  No pre-built KB found. Provide documents on your first .propose_experiments() call to create one.")

    def _build_and_save_kb(self, pdf_paths: List[str] = None, experimental_data: List[Dict[str, str]] = None) -> bool:
        """
        Internal method to parse documents, build the KB, and save it.
        Returns True on success, False on failure.
        """
        pdf_paths = pdf_paths or []
        experimental_data = experimental_data or []

        print("\n--- Parsing all provided documents ---")
        all_chunks = []
        for pdf_path in pdf_paths:
            all_chunks.extend(extract_pdf_two_pass(pdf_path))
        
        for data_pair in experimental_data:
            csv_chunk = parse_csv_with_context(data_pair.get('csv_path'), data_pair.get('context_path'))
            if csv_chunk:
                all_chunks.append(csv_chunk)

        if not all_chunks:
            print("❌ Build failed: No content extracted from the provided documents.")
            self._kb_is_built = False
            return False
            
        print("\n--- Building Knowledge Base (Embedding & Indexing) ---")
        self.knowledge_base.build(all_chunks)
        
        print("\n--- Saving Knowledge Base to Disk ---")
        self.knowledge_base.save(self.index_file, self.chunks_file)
        self._kb_is_built = True
        print(f"✅ Knowledge base built, saved to {self.kb_path_prefix}, and ready for queries.")
        return True

    def propose_experiments(self, objective: str, pdf_paths: List[str] = None, experimental_data: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generates experimental proposals. If documents are provided, it builds/rebuilds
        the knowledge base. Otherwise, it uses the existing/loaded KB.
        """
        # Decide whether to build/rebuild the KB
        new_documents_provided = (pdf_paths is not None) or (experimental_data is not None)

        if new_documents_provided:
            print("--- New documents provided. Building/rebuilding knowledge base... ---")
            build_success = self._build_and_save_kb(pdf_paths, experimental_data)
            if not build_success:
                return {"error": "Failed to build knowledge base from the provided documents."}
        elif not self._kb_is_built:
            # Error if no docs are provided AND no KB is loaded
            return {
                "error": "Knowledge base is not built. Please provide documents (e.g., 'pdf_paths') on your first call."
            }
        else:
             print("--- Using existing knowledge base. ---")

        # Retrieve relevant context
        print("\n--- Retrieving Relevant Context ---")
        context_chunks = self.knowledge_base.retrieve(objective, top_k=7)
        if not context_chunks:
            return {"error": "Could not retrieve any relevant context for the given objective."}

        # Construct prompt and generate hypothesis
        print("\n--- Generating Hypotheses with LLM ---")
        context_str = "\n\n---\n\n".join(
            f"Source: {Path(chunk['metadata']['source']).name}\n\n{chunk['text']}" 
            for chunk in context_chunks
        )
        prompt = f"""
        {HYPOTHESIS_GENERATION_INSTRUCTIONS}
        
        ## General Objective:
        {objective}
        
        ## Retrieved Context from Documents:
        {context_str}
        """
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            json_text = response.text.strip().lstrip("```json").rstrip("```")
            result = json.loads(json_text)
            print("  - ✅ Successfully generated and parsed experimental plan.")
            return result
        except Exception as e:
            raw_response_text = "N/A"
            if 'response' in locals() and hasattr(response, 'text'):
                raw_response_text = response.text
            return {"error": f"Failed to generate or parse LLM response: {str(e)}", "raw_response": raw_response_text}