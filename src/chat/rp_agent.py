from src.config import config, logger
from src.vector_store.vector_store_services import VectorStoreServices
from src.llm.query_processing import QueryProcessing
from src.llm.query_response import QueryResponse
from src.llm.reranking_chunks import RerankingChunks

class InferenceAgent:
    def __init__(self):
        self.vector_store_services = VectorStoreServices()
        self.query_processing = QueryProcessing()
        self.query_response = QueryResponse()
        self.reranking_chunks = RerankingChunks()

    def _agentic_flow(self, query: str) -> str:
        logger.info(f"Starting agentic flow for query: {query}")
        try:
            evaluation = self.query_processing._query_evaluation(query)

            if not evaluation:
                response = self.query_response._simple_query_response(query)

                if not response:
                    logger.error("Error in _agentic_flow: Query response failed")
                    return None
                
                return response

            queries = self.query_processing._rewrite_query(query, 3)

            logger.info(f"Rewritten queries: {queries}")

            if not queries:
                logger.error("Error in _agentic_flow: Query rewriting failed")
                return None
            
            retrieved_chunks = self.vector_store_services.retrieve_chunks(queries)

            if not retrieved_chunks:
                logger.warning("No chunks retrieved")
                return None

            if len(retrieved_chunks) == 0:
                logger.warning("No chunks retrieved")
                return None
            
            logger.info(f"Retrieved total of {len(retrieved_chunks)} chunks")

            context_string = "### CHUNKS\n"
            for chunk in retrieved_chunks:
                context_string += f"- {chunk.payload.page_content}\n"

            context = self.reranking_chunks._rerank_chunks_prepare_context(query, context_string, 'gpt-4o-mini')

            if not context:
                logger.error("Error in _agentic_flow: Context preparation failed")
                return None
            
            final_response = self.query_response._final_query_response(query, context, 'gpt-4o-mini')
            
            if not final_response:
                logger.error("Error in _agentic_flow: Final query response failed")
                return None
            
            logger.info("Final response generated successfully")

            return final_response               
        except Exception as e:
            logger.error(f"Error in _agentic_flow: {e}")
            raise Exception(f"Error in _agentic_flow: {e}")