import numpy as np
import faiss
import google.generativeai as genai
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper_embeddings import OpenAIAsEmbeddingModel

from openai import RateLimitError


class KnowledgeBase:
    """
    Handles embedding, retrieval, and repository structure mapping.
    Supports both Google and OpenAI-compatible (e.g., incubator) embedding models.
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
        
        # Registry for Repo Maps: {'repo_name': 'tree_structure_string'}
        # This stores the visual directory trees for any repo you ingest.
        self.repo_maps: Dict[str, str] = {}

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
            
            max_retries = 3
            delay = 5 # seconds
            for attempt in range(max_retries):
                try:
                    response = self.embedding_client.embed_content(
                        model=self.embedding_model_name,
                        content=batch_texts,
                        task_type="RETRIEVAL_DOCUMENT" # Ignored by OpenAI wrapper, used by Google
                    )
                    all_embeddings.extend(response['embedding'])
                    print(f"    - Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
                    time.sleep(1) # Small delay to respect API rate limits
                    break # Success
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        print(f"    - ⚠️  Rate limit hit during build. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2 # Exponential backoff
                    else:
                        print(f"    - ❌ Rate limit hit on final attempt. Build failed.")
                        raise e 
                except Exception as e:
                    print(f"    - ❌ Error embedding batch {i//batch_size + 1}: {e}")
                    raise e

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        
        print("  - Building FAISS vector index...")
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        print("  - ✅ Knowledge base built successfully.")

    def save(self, index_path: str, chunks_path: str, repo_map_path: str = None):
        """Saves the FAISS index, text chunks, and optionally the repo maps to disk."""
        if self.index:
            faiss.write_index(self.index, index_path)
            print(f"  - FAISS index saved to {index_path}")
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2)
            print(f"  - Chunks saved to {chunks_path}")

        # Save Repo Maps Registry
        if repo_map_path and self.repo_maps:
            try:
                with open(repo_map_path, 'w', encoding='utf-8') as f:
                    json.dump(self.repo_maps, f, indent=2)
                print(f"  - Repo maps registry saved to {repo_map_path}")
            except Exception as e:
                print(f"  - ❌ Error saving repo maps: {e}")

    def load(self, index_path: str, chunks_path: str, repo_map_path: str = None) -> bool:
        """Loads a pre-built FAISS index, chunks, and repo maps from disk."""
        index_file = Path(index_path)
        chunks_file = Path(chunks_path)

        if not index_file.exists() or not chunks_file.exists():
            print("  - ⚠️  Cannot load: Index or chunks file missing.")
            return False
            
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load Repo Maps if path provided and file exists
            if repo_map_path and Path(repo_map_path).exists():
                try:
                    with open(repo_map_path, 'r', encoding='utf-8') as f:
                        self.repo_maps = json.load(f)
                    print(f"    - Loaded maps for repos: {list(self.repo_maps.keys())}")
                except Exception as e:
                    print(f"    - ⚠️ Error loading repo maps file: {e}")
            
            print(f"  - ✅ Successfully loaded {len(self.chunks)} chunks and index with {self.index.ntotal} vectors.")
            return True
        except Exception as e:
            print(f"  - ❌ Error loading knowledge base: {e}")
            self.index = None
            self.chunks = []
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks for a given query.
        """
        if not self.index:
            print("⚠️  Cannot retrieve: Knowledge base has not been built.")
            return []
            
        print(f"  - Retrieving top {top_k} most relevant chunks for query: '{query[:80]}...'")

        max_retries = 3
        delay = 5 # seconds
        response = None
        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embed_content(
                    model=self.embedding_model_name,
                    content=query,
                    task_type="RETRIEVAL_QUERY" # Ignored by OpenAI wrapper, used by Google
                )
                break # Success
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"    - ⚠️  Rate limit hit embedding query. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
                else:
                    print(f"    - ❌ Rate limit hit on final attempt. Retrieval failed.")
                    raise e # Re-raise the exception if all retries fail
            except Exception as e:
                print(f"    - ❌ Error embedding query: {e}")
                raise e
        
        if response is None:
            print("    - ❌ Retrieval failed after retries.")
            return []

        query_embedding = np.array([response['embedding']], dtype=np.float32)

        if query_embedding.ndim == 3:
            query_embedding = np.squeeze(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve valid chunks (filtering out potential index errors)
        retrieved_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        print(f"  - ✅ Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks

    def get_relevant_maps(self, retrieved_chunks: List[Dict]) -> str:
        """
        Dynamic Context Injection:
        Looks at the retrieved chunks, finds which repos they belong to (via 'repo_name' metadata),
        and returns a combined string of ONLY the relevant repo maps.
        """
        relevant_repos = set()
        for chunk in retrieved_chunks:
            # We ensure chunks have this metadata field in planning_agent.py
            repo_name = chunk['metadata'].get('repo_name')
            if repo_name and repo_name in self.repo_maps:
                relevant_repos.add(repo_name)
        
        if not relevant_repos:
            return ""

        combined_map = ""
        for repo in relevant_repos:
            combined_map += f"\n--- DIRECTORY STRUCTURE FOR REPO: {repo} ---\n"
            combined_map += self.repo_maps[repo]
            combined_map += "\n"
            
        return combined_map