from src.config import config, logger
from src.document_ingestion.extraction import DocumentExtractor
from src.document_ingestion.chunking import chunking
from src.vector_store.vector_store_services import VectorStoreServices

class DocumentIngestionPipeline:
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.chunking = chunking()
        self.vector_store_services = VectorStoreServices()

    def _forward_pipeline(self):
        logger.info("Document extraction and ingestion pipeline for user documents")
        try:
            extracted_docs = self.document_extractor._extract_documents_content()

            if len(extracted_docs) == 0:
                logger.warning("No documents extracted")
                return False
            
            self.vector_store_services.create_vector_store()
            
            for extracted_doc in extracted_docs:
                chunks = self.chunking.split_page(extracted_doc)

                if len(chunks) == 0:
                    logger.warning(f"No chunks extracted from {extracted_doc}")
                    continue
                
                result = self.vector_store_services.upsert_chunks(chunks)

                if not result:
                    logger.error(f"Error upserting chunks for {extracted_doc}")
                    continue

                logger.info(f"Upserted {len(chunks)} chunks into vector store")


            return True
        except Exception as e:
            logger.error(f"Error in document extraction and ingestion pipeline: {e}")
            raise Exception(f"Error in document extraction and ingestion pipeline: {e}")
        