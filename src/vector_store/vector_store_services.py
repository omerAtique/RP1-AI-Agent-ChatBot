from src.config import config, logger
from src.document_ingestion.extraction import ExtractedDocument
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.schema import Document
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams


class VectorStoreServices:
    def __init__(self):
        self.client = QdrantClient(
            url=config.lc_qdrant.qdrant_url,
            api_key=config.lc_qdrant.qdrant_api_key
        )

        self.dense_embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self.vector_store = QdrantVectorStore(  
                    client=self.client,
                    collection_name=config.lc_qdrant.collection_name,
                    embedding=self.dense_embedding,
                    sparse_embedding=self.sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    vector_name="dense",
                    sparse_vector_name="sparse"
                )

        
    def create_vector_store(self):
        logger.info("Creating vector store")
        try:

            if self.client.collection_exists(config.lc_qdrant.collection_name):
                logger.info(f"Vector store {config.lc_qdrant.collection_name} already exists")

                return True

            self.client.create_collection(
                collection_name=config.lc_qdrant.collection_name,
                vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )

            if self.vector_store.client.collection_exists(config.lc_qdrant.collection_name):
                logger.info(f"Vector store {config.lc_qdrant.collection_name} created successfully")
                return True
            else:
                logger.error(f"Vector store {config.lc_qdrant.collection_name} creation failed")
                return False
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise Exception(f"Error creating vector store: {e}")
    
    def upsert_chunks(self, chunks: List[Document]):
        try:
            if len(chunks) == 0:
                logger.warning("No chunks to upsert")
                return False
            
            MAX_BATCH_SIZE = 200
            for i in range(0, len(chunks), MAX_BATCH_SIZE):
                try:
                    batch = chunks[i:i+MAX_BATCH_SIZE]
                    self.vector_store.add_documents(batch)
                except Exception as e:
                    logger.error(f"Error upserting batch {i}: {e}")
                    raise Exception(f"Error upserting batch {i}: {e}")

            logger.info(f"Upserted {len(chunks)} chunks")

            return True
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            raise Exception(f"Error upserting chunks: {e}")
    
    def retrieve_chunks(self, queries: List[str], k: int = 15):
        logger.info(f"Retrieving chunks for {len(queries)} queries")
        try:
            prefetch_queries = []
        
            for query_text in queries:

                dense_embedding = self.dense_embedding.embed_query(query_text)
                sparse_embedding = self.sparse_embeddings.embed_query(query_text)
                
                if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
                    sparse_query = {
                        "indices": sparse_embedding.indices,
                        "values": sparse_embedding.values
                    }
                else:
                    sparse_query = sparse_embedding
                
                prefetch_queries.append(
                    models.Prefetch(
                        query=dense_embedding,
                        using="dense",
                        limit=k,
                    )
                )
                
                prefetch_queries.append(
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse", 
                        limit=k,
                    )
                )

            results = self.client.query_points(
                collection_name=config.lc_qdrant.collection_name,
                prefetch=prefetch_queries,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
            )

            if len(results.points) == 0:
                logger.warning("No results found")
                return []
            else:
                logger.info(f"Retrieved {len(results.points)} chunks")
                formatted_chunks = []
                for point in results.points:
                    chunk_obj = type('Chunk', (), {
                        'payload': type('Payload', (), {
                            'page_content': point.payload.get('page_content', '')
                        })()
                    })()
                    formatted_chunks.append(chunk_obj)
                
                return formatted_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise Exception(f"Error retrieving chunks: {e}")
        
    def delete_vector_store(self):
        logger.info("Deleting vector store")
        try:
            deleted = self.client.delete_collection(config.lc_qdrant.collection_name)

            if deleted:
                logger.info(f"Vector store {config.lc_qdrant.collection_name} deleted")
                return True
            else:
                logger.warning(f"Vector store {config.lc_qdrant.collection_name} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            raise Exception(f"Error deleting vector store: {e}")
