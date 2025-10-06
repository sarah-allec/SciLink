import numpy as np
import faiss
import google.generativeai as genai
import time
from typing import List, Dict, Any
from ...auth import get_api_key, APIKeyNotFoundError

class KnowledgeBase:
    """
    Handles the embedding, storage, and retrieval of document chunks for RAG.
    This version is compatible with embedding models that do not use the 'task_type'
    parameter, such as 'gemini-embedding-001' or 'models/embedding-001'.
    """
    def __init__(self, google_api_key: str = None, embedding_model: str = "gemini-embedding-001"):
        """
        Initializes the KnowledgeBase.
        
        Args:
            google_api_key (str, optional): Your Google API key. Defaults to auto-discovery.
            embedding_model (str, optional): The name of your embedding model. 
                                           Defaults to "gemini-embedding-001".
        """
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        genai.configure(api_key=google_api_key)
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build(self, chunks: List[Dict[str, any]], batch_size: int = 100):
        """
        Processes a list of text chunks, generates embeddings in batches, 
        and builds the vector index.
        """
        if not chunks:
            print("⚠️ KnowledgeBase build skipped: No chunks provided.")
            return

        self.chunks = chunks
        texts_to_embed = [chunk['text'] for chunk in self.chunks]
        all_embeddings = []
        
        print(f"  - Generating embeddings for {len(texts_to_embed)} chunks using '{self.embedding_model}'...")
        
        # Process in batches to respect API limits and manage memory
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            
            # Generate embeddings - NOTE: No 'task_type' or 'title' is used.
            response = genai.embed_content(
                model=self.embedding_model,
                content=batch_texts,
            )
            all_embeddings.extend(response['embedding'])
            
            print(f"    - Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
            time.sleep(1) # Small delay to respect rate limits

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

        # Embed the query - NOTE: No 'task_type' is used.
        response = genai.embed_content(
            model=self.embedding_model,
            content=query,
        )
        query_embedding = np.array([response['embedding']], dtype=np.float32)

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return the original chunks corresponding to the retrieved indices
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        print(f"  - ✅ Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks