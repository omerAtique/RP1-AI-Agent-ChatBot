from src.config import config, logger, openai_client
from pydantic import BaseModel
from src.prompts.query_processing_prompts import query_evaluation_prompt, query_rewriting_prompt
import json

class QueryEvaluationResponse(BaseModel):
    query_evaluation: bool

class QueryRewriteResponse(BaseModel):
    rewrittenQueries: list[str]

class QueryProcessing:
    def __init__(self):
        self.openai_client = openai_client
    
    def _query_evaluation(self, query: str, model: str = 'gpt-4o-mini') -> bool:
        logger.info(f"Evaluating query: {query}")
        try:
            response = self.openai_client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query_evaluation_prompt(query)}
                ],
                temperature=0.1,
                text_format=QueryEvaluationResponse
            )

            if not response:
                logger.error("Error in query_evaluation: No response from the model")
                return None
            
            response = response.output_parsed

            if not response:
                logger.error("Error in query_evaluation: No response from the model")
                return None
            
            evaluation: bool = response.query_evaluation
            
            return evaluation
        except Exception as e:
            logger.error(f"Error in query_evaluation: {e}")
            raise Exception(f"Error in query_evaluation: {e}")
        
    def _rewrite_query(self, query: str, num_queries: int, model: str = 'gpt-4o-mini') -> list[str]:
        logger.info(f"Rewriting query: {query} into {num_queries} queries")
        try:
            response = self.openai_client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query_rewriting_prompt(query, num_queries)}
                ],
                temperature=0.1,
                text_format=QueryRewriteResponse
            )

            if not response:
                logger.error("Error in rewrite_query: No response from the model")
                return None
            
            response = response.output_parsed

            if not response:
                logger.error("Error in rewrite_query: No response from the model")
                return None

            queries: list[str] = response.rewrittenQueries
            return queries
        except Exception as e:
            logger.error(f"Error in rewrite_query: {e}")
            raise Exception(f"Error in rewrite_query: {e}")

        






