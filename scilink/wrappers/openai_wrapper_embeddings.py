import openai
from typing import List

class OpenAIAsEmbeddingModel:
    """
    Mimics Google's genai.embed_content function using an OpenAI-compatible API.
    """
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_content(self, model: str, content: List[str], task_type: str = None, title: str = None, **kwargs) -> dict:
        """
        Generates embeddings for a list of text content.
        
        Args:
            model (str): The model name (ignored, uses model from __init__).
            content (List[str]): A list of strings to embed.
            task_type (str, optional): Ignored for compatibility.
            title (str, optional): Ignored for compatibility.

        Returns:
            dict: A dictionary in the format {'embedding': [vector1, vector2, ...]}
                  to match the google-generativeai API response.
        """
        # OpenAI-compatible API uses the 'input' parameter
        response = self.client.embeddings.create(
            model=self.model,
            input=content
        )
        
        # Transform the OpenAI-style response to the Google-style response
        # OpenAI: response.data = [Embedding(embedding=[...]), Embedding(embedding=[...])]
        # Google: response = {'embedding': [[...], [...]]}
        all_embedding_vectors = [item.embedding for item in response.data]
        
        return {'embedding': all_embedding_vectors}