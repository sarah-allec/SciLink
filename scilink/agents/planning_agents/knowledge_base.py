import numpy as np
import faiss
import google.generativeai as genai
import time
import logging
from typing import List, Dict, Any

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper_embeddings import OpenAIAsEmbeddingModel

class KnowledgeBase:
    """
    Handles embedding and retrieval. Supports both Google and 
    OpenAI-compatible (e.g., incubator) embedding models.
    """
    def __init__(self, google_api_key: str = None, 
                 embedding_model: str = "gemini-embedding-001", 
                 local_model: str = None):
        
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        self.embedding_model_name = embedding_model
        
        # --- Logic to Switch Embedding Backends ---
        if local_model and 'ai-incubator' in local_model:
            logging.info(f"🏛️  Using OpenAI-compatible incubator model for embeddings: {self.embedding_model_name}")
            self.embedding_client = OpenAIAsEmbeddingModel(
                model=self.embedding_model_name,
                api_key=google_api_key, # This key is for the incubator service
                base_url=local_model
            )
        else:
            logging.info(f"☁️  Using Google Gemini model for embeddings: {self.embedding_model_name}")
            # For Google, the client is the genai module itself after configuration
            genai.configure(api_key=google_api_key)
            self.embedding_client = genai
            
        self.index = None
        self.chunks = []

    def build(self, chunks: List[Dict[str, any]], batch_size: int = 100):
        """
        Processes a list of text chunks, generates embeddings in batches, 
        and builds the vector index.
        """
        if not chunks:
            print("⚠️  KnowledgeBase build skipped: No chunks provided.")
            return

        self.chunks = chunks
        texts_to_embed = [chunk['text'] for chunk in self.chunks]
        all_embeddings = []
        
        print(f"  - Generating embeddings for {len(texts_to_embed)} chunks using '{self.embedding_model_name}'...")
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            
            # This call is polymorphic and works with either backend
            response = self.embedding_client.embed_content(
                model=self.embedding_model_name,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT" # Ignored by OpenAI wrapper, used by Google
            )
            all_embeddings.extend(response['embedding'])
            print(f"    - Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
            time.sleep(1) # Small delay to respect API rate limits

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        
        print("  - Building FAISS vector index...")
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        print("  - ✅ Knowledge base built successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks for a given query.
        """
        if not self.index:
            print("⚠️  Cannot retrieve: Knowledge base has not been built.")
            return []
            
        print(f"  - Retrieving top {top_k} most relevant chunks for query: '{query[:80]}...'")

        # This call is also polymorphic
        response = self.embedding_client.embed_content(
            model=self.embedding_model_name,
            content=query,
            task_type="RETRIEVAL_QUERY" # Ignored by OpenAI wrapper, used by Google
        )
        query_embedding = np.array([response['embedding']], dtype=np.float32)

        if query_embedding.ndim == 3:
            query_embedding = np.squeeze(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        print(f"  - ✅ Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks