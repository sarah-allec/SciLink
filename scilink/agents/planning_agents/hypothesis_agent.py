import google.generativeai as genai
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass
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
        Initializes the agent, LLM backends, and attempts to load the
        knowledge base from the specified disk location.
        """
        
        if google_api_key is None:
            # Note: This key will be used for the incubator model if local_model is set
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        # --- Logic to Switch Generation LLM Backend ---
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  Using OpenAI-compatible model for generation: {model_name}")
            self.model = OpenAIAsGenerativeModel(
                model_name,
                api_key=google_api_key,
                base_url=local_model
            )
            self.generation_config = None 
        else:
            logging.info(f"☁️  Using Google Gemini model for generation: {model_name}")
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        # Configure the KnowledgeBase, passing the backend choice to it
        self.knowledge_base = KnowledgeBase(
            google_api_key=google_api_key,
            embedding_model=embedding_model,
            local_model=local_model # Pass the incubator URL down
        )

        # Ensure the directory for the KB exists
        self.kb_path_prefix = Path(kb_base_path)
        self.kb_path_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Define the two file paths from the single base path
        self.index_file = str(self.kb_path_prefix.with_suffix(".faiss"))
        self.chunks_file = str(self.kb_path_prefix.with_suffix(".json"))
        
        # Attempt to load KB on initialization
        print(f"--- Initializing Agent ---")
        print(f"Attempting to load Knowledge Base from: {self.kb_path_prefix}.(faiss/json)")
        self._kb_is_built = self.knowledge_base.load(
            self.index_file, 
            self.chunks_file
        )
        if self._kb_is_built:
            print("✅ Knowledge base loaded successfully.")
        else:
            print(f"⚠️  No pre-built KB found.")
            print("   Call .build_knowledge_base(pdf_paths) to create one.")


    def build_knowledge_base(self, pdf_paths: List[str]):
        """
        Parses PDFs, builds the vector index, and saves it to the
        disk location specified during initialization.
        """
        # Parse all PDFs into chunks
        print("\n--- Parsing PDF documents ---")
        all_chunks = []
        for pdf_path in pdf_paths:
            all_chunks.extend(extract_pdf_two_pass(pdf_path))
        
        if not all_chunks:
            print("❌ Error: Failed to extract any content from the provided PDFs.")
            self._kb_is_built = False
            return False
            
        # Build the knowledge base from the chunks
        print("\n--- Building Knowledge Base (Embedding & Indexing) ---")
        self.knowledge_base.build(all_chunks)
        
        # Save the built KB to disk
        print(f"\n--- Saving Knowledge Base to Disk ---")
        self.knowledge_base.save(self.index_file, self.chunks_file)

        self._kb_is_built = True
        print(f"✅ Knowledge base is built, saved to {self.kb_path_prefix}, and ready.")
        return True

    def propose_experiments(self, objective: str, pdf_paths: List[str] = None) -> Dict[str, Any]:
        """
        Generates experimental proposals based on the loaded knowledge base.
        
        Assumes build_knowledge_base() has been called if the KB was not
        already pre-built and loaded during initialization.
        """
        # Load-or-Build check
        if not self._kb_is_built:
            if pdf_paths:
                print(f"⚠️  Knowledge base not built. Building from provided PDFs...")
                build_success = self.build_knowledge_base(pdf_paths)
                if not build_success:
                    return {"error": "Failed to build knowledge base from provided PDFs."}
            else:
                return {
                    "error": "Knowledge base is not built. Please provide 'pdf_paths' on the first call, or call .build_knowledge_base() first."
                }

        # Retrieve relevant context based on the user's objective
        print("\n--- Retrieving Relevant Context ---")
        context_chunks = self.knowledge_base.retrieve(objective, top_k=7)
        if not context_chunks:
            return {"error": "Could not retrieve any relevant context from the documents for the given objective."}

        # Construct the prompt and generate the hypothesis
        print("\n--- Generating Hypotheses with LLM ---")
        context_str = "\n\n---\n\n".join(
            f"Source: {Path(chunk['metadata']['source']).name}, Page: {chunk['metadata']['page']}\n\n{chunk['text']}" 
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