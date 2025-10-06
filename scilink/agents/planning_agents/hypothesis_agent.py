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
    Supports both Google and OpenAI-compatible (e.g., incubator) models.
    """
    def __init__(self, google_api_key: str = None, 
                 model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 embedding_model: str = "gemini-embedding-001"):
        
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
                api_key=google_api_key, # The key for the incubator/OpenAI service
                base_url=local_model   # The URL for the incubator/OpenAI service
            )
            self.generation_config = None 
        else:
            logging.info(f"☁️  Using Google Gemini model for generation: {model_name}")
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        # --- Configure the KnowledgeBase, passing the backend choice to it ---
        self.knowledge_base = KnowledgeBase(
            google_api_key=google_api_key,
            embedding_model=embedding_model,
            local_model=local_model # Pass the incubator URL down to the knowledge base
        )

    def propose_experiments(self, objective: str, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        The main method to run the full RAG pipeline and generate experimental proposals.
        """
        # Step 1: Parse all PDFs into chunks
        print("\n--- Step 1: Parsing PDF documents ---")
        all_chunks = []
        for pdf_path in pdf_paths:
            all_chunks.extend(extract_pdf_two_pass(pdf_path))
        
        if not all_chunks:
            return {"error": "Failed to extract any content from the provided PDFs."}
            
        # Step 2: Build the knowledge base from the chunks
        print("\n--- Step 2: Building Knowledge Base ---")
        self.knowledge_base.build(all_chunks)
        
        # Step 3: Retrieve relevant context based on the user's objective
        print("\n--- Step 3: Retrieving Relevant Context ---")
        context_chunks = self.knowledge_base.retrieve(objective, top_k=7)
        if not context_chunks:
            return {"error": "Could not retrieve any relevant context from the documents for the given objective."}

        # Step 4: Construct the prompt and generate the hypothesis
        print("\n--- Step 4: Generating Hypotheses with LLM ---")
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