import asyncio
from typing import List, Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class AIEnsembleClient:
    def __init__(self, llm_clients: List[LLMClient]):
        self.llm_clients = llm_clients
        self.prompt_renderer = PromptRenderer()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_llm_call(self, client: LLMClient, model: str, prompt: str, temperature: float) -> Any:
        try:
            # 假设 client.generate_text 是存在的
            return await client.generate_text(model, prompt, temperature)
        except Exception as e:
            logger.error(f"Error executing LLM call for client {client.client_name}: {e}")
            return None

    async def execute_map_reduce(self, long_texts: List[str], original_query: str) -> str:
        # Map Phase
        map_tasks = []
        all_chunks = []
        for long_text in long_texts:
            all_chunks.extend(self.prompt_renderer.chunk_context(long_text))

        for i, chunk in enumerate(all_chunks):
            client = self.llm_clients[i % len(self.llm_clients)]
            prompt = f"Based on the following text: '{chunk}', answer this question: '{original_query}'."
            model = client.get_models()[0] if client.get_models() else "default-model"
            map_tasks.append(
                self.execute_llm_call(client, model, prompt, temperature=0.5)
            )

        sub_conclusions_results = await asyncio.gather(*map_tasks)

        # Reduce Phase
        successful_sub_conclusions = [res for res in sub_conclusions_results if res is not None]

        if not successful_sub_conclusions:
            return "Could not generate a final conclusion as all sub-tasks failed."

        reducer_prompt = self.prompt_renderer.render_reducer_prompt(original_query, successful_sub_conclusions)
        
        final_client = self.llm_clients[0]
        final_model = final_client.get_models()[0] if final_client.get_models() else "default-model"
        final_conclusion = await self.execute_llm_call(final_client, final_model, reducer_prompt, temperature=0.7)

        return final_conclusion

    async def execute_concurrent_calls(self, prompts: List[Dict[str, Any]]) -> List[Any]:
        tasks = []
        for i, prompt_data in enumerate(prompts):
            client = self.llm_clients[i % len(self.llm_clients)]
            tasks.append(
                self.execute_llm_call(
                    client,
                    prompt_data["model"],
                    prompt_data["prompt"],
                    prompt_data["temperature"]
                )
            )
        return await asyncio.gather(*tasks)
