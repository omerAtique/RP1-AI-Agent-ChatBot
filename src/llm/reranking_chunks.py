from src.config import config, logger, openai_client
from pydantic import BaseModel
from src.prompts.reranking_chunks_prompt import reranking_chunks_prompt
import json

class RerankedContext(BaseModel):
    context: str

class RerankingChunks:
    def __init__(self):
        self.openai_client = openai_client

    def _rerank_chunks_prepare_context(self, query: str, context: str, model: str = 'gpt-4.1-mini') -> str:
        logger.info(f"Preparing context for final response for query: {query}")
        try:
            response = self.openai_client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": reranking_chunks_prompt(query, context)}
                ],
                temperature=0.1,
                text_format=RerankedContext
            )

            response = response.output_parsed

            if not response:
                logger.error("Error in _rerank_chunks_prepare_context: No response from the model")
                return None

            context: str = response.context
            return context
        except Exception as e:
            logger.error(f"Error in _rerank_chunks_prepare_context: {e}")
            raise Exception(f"Error in _rerank_chunks_prepare_context: {e}")

