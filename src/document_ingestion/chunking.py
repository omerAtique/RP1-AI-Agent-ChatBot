from langchain_text_splitters import TokenTextSplitter
from src.document_ingestion.extraction import ExtractedDocument
from typing import List
from src.config import logger

text_splitter = TokenTextSplitter(
    encoding_name="o200k_base", chunk_size=400, chunk_overlap=50
)


class chunking:
    def __init__(self):
        self.text_splitter = text_splitter

    def split_page(self, page: List[ExtractedDocument]):
        try:
            page_contents = [page["fullText"] for page in page]
            metadata = [{"documentName": page["documentName"], "pageNumber": page["pageNumber"]} for page in page]
            chunks = self.text_splitter.create_documents(page_contents, metadata)
            logger.info(f"Split pages into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error splitting page: {e}")
            raise Exception(f"Error splitting page: {e}")






