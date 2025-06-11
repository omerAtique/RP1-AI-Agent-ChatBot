from src.config import config, logger, openai_client
from pydantic import BaseModel
from src.prompts.query_response_prompts import non_retrieval_response_prompt, final_query_response_prompt
import json

class QueryResponseFormat(BaseModel):
    query_response: str

class QueryResponse:
    def __init__(self):
        self.openai_client = openai_client

    def _simple_query_response(self, query: str, model: str = 'gpt-4o-mini') -> str:
        logger.info(f"Generating simple query response for query: {query}")
        try:
            response = self.openai_client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": non_retrieval_response_prompt(query)}
                ],
                temperature=0.1,
                text_format=QueryResponseFormat
            )

            response = response.output_parsed

            if not response:
                logger.error("Error in _simple_query_response: No response from the model")
                return None

            query_response: str = response.query_response
            return query_response
        except Exception as e:
            logger.error(f"Error in _query_response: {e}")
            raise Exception(f"Error in _query_response: {e}")
        
    def _final_query_response(self, query: str, context: str, model: str = 'gpt-4.1-mini') -> str:
        logger.info(f"Generating final query response for query: {query}")
        try:
            response = self.openai_client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": final_query_response_prompt(query, context)}
                ],
                temperature=0.1,
                text_format=QueryResponseFormat
            )
            
            response = response.output_parsed
            
            if not response:
                logger.error("Error in _final_query_response: No response from the model")
                return None
            
            query_response: str = response.query_response
            return query_response
        except Exception as e:
            logger.error(f"Error in _final_query_response: {e}")
            raise Exception(f"Error in _final_query_response: {e}")
        
            

