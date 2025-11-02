import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from api.gemini_pool_manager import GeminiPoolManager
from monitor.logging import get_logger
from config.loader import ConfigLoader

logger = get_logger(__name__)

class APIGateway:
    """
    Main gateway for all external API calls, primarily to Google Gemini.
    Manages API key, model selection, request formatting, and error handling.
    Uses GeminiPoolManager for concurrent requests.
    """

    def __init__(self, config_loader: ConfigLoader, pool_size: int = 10):
        api_key = config_loader.get_secret("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment.")
            raise ValueError("GEMINI_API_KEY is not set.")
        
        genai.configure(api_key=api_key)
        
        self.pool_manager = GeminiPoolManager(
            api_key=api_key,
            pool_size=pool_size
        )
        # TODO: Load safety settings and generation config from config_loader
        self.default_safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        self.default_generation_config = {
            "candidate_count": 1,
            # "stop_sequences": ["..."], # Example
            "max_output_tokens": 4096,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 1,
        }
        
        logger.info(f"APIGateway initialized with pool size {pool_size}.")

    async def send_request(
        self,
        model_name: str,
        prompt: Union[str, List[Union[str, Any]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Sends a request to the Gemini API using the pool manager.
        
        Args:
            model_name (str): The name of the model (e.g., "gemini-pro").
            prompt (Union[str, List[Union[str, Any]]]): The prompt string or
                                                       a list of content parts.
            
        Returns:
            str: The text response from the model.
        """
        
        # Override defaults if provided
        generation_config = self.default_generation_config.copy()
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_config["stop_sequences"] = stop_sequences
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
            
        # Format the prompt content
        contents = [prompt] if isinstance(prompt, str) else prompt

        try:
            logger.debug(f"Sending request to model '{model_name}' via pool...")
            response_text = await self.pool_manager.generate_content(
                model_name=model_name,
                contents=contents,
                generation_config=generation_config,
                safety_settings=self.default_safety_settings
            )
            logger.debug(f"Received response from '{model_name}'.")
            return response_text
        except Exception as e:
            logger.error(f"Error in send_request via pool for '{model_name}': {e}", exc_info=True)
            # Re-raise or return a specific error message
            raise

    async def send_embedding_request(
        self,
        model_name: str,
        texts: List[str],
        dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Sends a batch embedding request.
        
        Args:
            model_name (str): The embedding model name (e.g., "text-embedding-3-large").
            texts (List[str]): List of texts to embed.
            dimensions (Optional[int]): Requested embedding dimensions.
            
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        
        # This uses the genai library directly, as the pool manager
        # is set up for `generate_content`. We could extend the pool manager
        # to also handle `embed_content`.
        
        # For now, use asyncio.to_thread for the blocking SDK call
        
        def blocking_embed_call():
            try:
                logger.debug(f"Embedding {len(texts)} texts with model '{model_name}'...")
                # Note: `output_dimensionality` is the arg name in genai
                result = genai.embed_content(
                    model=f"models/{model_name}",
                    content=texts,
                    task_type="retrieval_document", # Or "semantic_similarity" etc.
                    output_dimensionality=dimensions
                )
                logger.debug("Embedding call successful.")
                return result['embedding']
            except Exception as e:
                logger.error(f"Error during blocking embed call: {e}", exc_info=True)
                raise

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, blocking_embed_call)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return [[] for _ in texts]


# Example usage (if run directly, though it's not typical)
if __name__ == "__main__":
    # This requires a valid config setup
    logger.warning("APIGateway is not intended to be run directly.")
    
    # Example async test
    async def test_gateway():
        try:
            # Mock ConfigLoader for testing
            class MockConfigLoader:
                def get_secret(self, key):
                    import os
                    return os.environ.get(key) # Assumes GEMINI_API_KEY is set in env
            
            config_loader = MockConfigLoader()
            gateway = APIGateway(config_loader, pool_size=2)
            
            prompt = "What is the capital of France?"
            response = await gateway.send_request("gemini-pro", prompt, temperature=0.0)
            print(f"Test Request 1:\nPrompt: {prompt}\nResponse: {response}\n")

            prompt_2 = "Explain the concept of asynchronous programming in Python."
            response_2 = await gateway.send_request("gemini-pro", prompt_2, max_tokens=150)
            print(f"Test Request 2:\nPrompt: {prompt_2}\nResponse: {response_2}\n")
            
            # Test embedding
            texts = ["Hello world", "This is a test"]
            embeddings = await gateway.send_embedding_request("text-embedding-3-large", texts, dimensions=256)
            print(f"Test Embedding:\nTexts: {texts}\nEmbedding 1 dim: {len(embeddings[0])}\n")

        except Exception as e:
            print(f"Error during test: {e}")

    # asyncio.run(test_gateway()) # Uncomment to run test
