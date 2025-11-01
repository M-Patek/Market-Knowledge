from typing import Dict, Any, List, Optional
import asyncio

from ..memory.vector_store import VectorStore
from ..api.gemini_pool_manager import GeminiPoolManager
from ..ai.prompt_manager import PromptManager

class FactChecker:
    """
    Responsible for verifying factual claims made by other agents.
    It uses a separate vector store (or could use Google Search) to
    find supporting or contradictory evidence for a given claim.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        gemini_pool: GeminiPoolManager,
        prompt_manager: PromptManager,
        vector_store: VectorStore
    ):
        """
        Initializes the FactChecker.
        
        Args:
            config: Main configuration object.
            gemini_pool: Pool for accessing Gemini models.
            prompt_manager: To get the 'fact_checker' prompt.
            vector_store: Vector store client (e.g., Pinecone) to search for evidence.
        """
        self.config = config.get('fact_checker', {})
        self.gemini_pool = gemini_pool
        self.prompt_manager = prompt_manager
        self.vector_store = vector_store
        
        self.model_id = self.config.get('model_id', 'gemini-1.5-pro')
        self.top_k_evidence = self.config.get('top_k_evidence', 3)
        
    async def check_claim(self, claim: str, event_id: str) -> Dict[str, Any]:
        """
        Checks a single factual claim (e.g., "AAPL EPS beat estimates").
        
        Args:
            claim (str): The statement to verify.
            event_id (str): The ID of the event for logging/tracing.
            
        Returns:
            Dict[str, Any]: A structured response, e.g.,
            {
                "claim": claim,
                "status": "VERIFIED" | "CONTRADICTED" | "NOT_FOUND",
                "confidence": 0.9,
                "evidence": [...]
            }
        """
        
        # 1. Retrieve evidence
        try:
            # Use the vector store to find relevant documents
            evidence_chunks = await self.vector_store.search(
                query_text=claim,
                top_k=self.top_k_evidence
                # TODO: Add metadata filters (e.g., time range)
            )
        except Exception as e:
            return {
                "claim": claim,
                "status": "ERROR",
                "confidence": 0.0,
                "justification": f"Failed to retrieve evidence: {e}",
                "evidence": []
            }
            
        if not evidence_chunks:
            return {
                "claim": claim,
                "status": "NOT_FOUND",
                "confidence": 0.0,
                "justification": "No supporting evidence found in vector store.",
                "evidence": []
            }

        # 2. Format context for the LLM
        context_str = "\n\n---\n\n".join(
            [f"Source: {chunk.get('metadata', {}).get('url', 'Unknown')}\n{chunk.get('text')}" 
             for chunk in evidence_chunks]
        )
        
        # 3. Get prompts
        system_prompt = self.prompt_manager.get_system_prompt('fact_checker')
        # The 'fact_checker' user prompt template is assumed to take 'claim' and 'evidence'
        user_prompt_template = self.prompt_manager.get_user_prompt_template('fact_checker') 
        user_prompt = user_prompt_template.format(claim=claim, evidence=context_str)
        
        # 4. Call LLM to verify
        try:
            async with self.gemini_pool.get_client(self.model_id) as gemini_client:
                response = await gemini_client.generate_content_async(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    request_id=f"{event_id}_fact_check",
                    generation_config={"response_mime_type": "application/json"}
                )
            
            # The response is expected to be a JSON object like the return dict
            # Add the original claim back in
            response['claim'] = claim
            
            # TODO: Add validation on the response schema
            
            return response

        except Exception as e:
            return {
                "claim": claim,
                "status": "ERROR",
                "confidence": 0.0,
                "justification": f"LLM verification failed: {e}",
                "evidence": [chunk.get('metadata', {}) for chunk in evidence_chunks]
            }

    async def batch_check_claims(self, claims: List[str], event_id: str) -> List[Dict[str, Any]]:
        """
        Checks a list of factual claims in parallel.
        
        Args:
            claims (List[str]): List of claims to verify.
            event_id (str): The base event ID.
            
        Returns:
            List[Dict[str, Any]]: A list of verification results.
        """
        tasks = [self.check_claim(claim, f"{event_id}_claim_{i}") for i, claim in enumerate(claims)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                final_results.append({
                    "claim": claims[i],
                    "status": "ERROR",
                    "confidence": 0.0,
                    "justification": f"Batch check task failed: {res}",
                    "evidence": []
                })
            else:
                final_results.append(res)
                
        return final_results
