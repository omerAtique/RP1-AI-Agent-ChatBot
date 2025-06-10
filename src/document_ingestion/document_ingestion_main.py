from src.document_ingestion.pipeline import DocumentIngestionPipeline
from src.config import config, logger

def main():
    logger.info("Starting document ingestion pipeline")
    try:
        document_ingestion_pipeline = DocumentIngestionPipeline()

        result = document_ingestion_pipeline._forward_pipeline()

        if not result:
            logger.error("Document ingestion pipeline failed")
            return False
        
        logger.info("Document ingestion pipeline completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in document ingestion pipeline: {e}")
        raise Exception(f"Error in document ingestion pipeline: {e}")
    

if __name__ == "__main__":
    main()