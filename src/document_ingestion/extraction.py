from src.config import config, logger
from mistralai import Mistral
import os
import base64
from pydantic import BaseModel

class ExtractedDocument(BaseModel):
    fullText: str
    pageNumber: int

class DocumentExtractor:
    def __init__(self):
        self.mistral = config.mistral
        self.mistral_client = Mistral(
            api_key=self.mistral.api_key,
            retry_config={
                "strategy": "backoff",
                "backoff": {
                    "initialInterval": 500,
                    "maxInterval": 60000,
                    "exponent": 1.5,
                    "maxElapsedTime": 300000
                },
                "retryConnectionErrors": True
            }
            )
        self.documents = []
        self.doc_folder_path = config.etl.docs_folder_path

    def _extract_documents_content(self):
        logger.info(f"Extracting documents from {self.doc_folder_path}")
        try:
            for file in os.listdir(self.doc_folder_path):
                if file.endswith(".pdf") or file.endswith(".docx"):
                    logger.info(f"Extracting document {file}")
                    self.documents.append(file)
                else:
                    logger.warning(f"Skipping file {file} as it is not a PDF or DOCX file")
        
            # split the total documents into batches of 6 documents. Files in 1 batch are processed in parallel with mistral
            extracted_docs = []
            for doc in self.documents:
                doc_path = os.path.join(self.doc_folder_path, doc)
                
                extracted_content = self._extraction_using_mistral(doc_path)
                if extracted_content:
                    logger.info(f"Extracted content from {doc} successfully")
                else:
                    logger.error(f"Error extracting content from {doc}")


                extracted_docs.append(extracted_content)

            logger.info(extracted_docs[0])

            return extracted_docs
        except Exception as e:
            logger.error(f"Error extracting documents: {e}")
            raise Exception(f"Error extracting documents: {e}")

    def _extraction_using_mistral(self, path: str):
        logger.info(f"Extracting content from document")
        try:
            doc_path = os.path.basename(path)

            with open(path, "rb") as doc_file:
                doc = doc_file.read()

            uploadResult = self.mistral_client.files.upload(
                file={
                        "fileName": doc_path,
                        "content": doc
                    },
                purpose="ocr"
            )

            fileSignedURL = self.mistral_client.files.get_signed_url(
                file_id=uploadResult.id
            )

            response = self.mistral_client.ocr.process(
                document={
                    "document_url": fileSignedURL.url,
                    "document_name": doc_path,
                    "type": "document_url"
                    },
                model="mistral-ocr-latest",
                timeout_ms=12000,
                retries=3,
                include_image_base64=True
            )

            logger.info(f"File uploaded successfully")

            if not response:
                logger.error(f"Error extracting content from {doc_path}: {response.error}")
                return None
            
            if not response.pages or len(response.pages) == 0:
                logger.error(f"No pages found in OCR response")
                
            else:
                logger.info(f"Extracted {len(response.pages)} pages from {doc_path}")

            logger.info(f"OCR response recieved successfully")

            structured_content_array = [
                {
                    "fullText": page.markdown,
                    "pageNumber": page.index + 1,
                }
                
                for page in response.pages
            ]

            [ExtractedDocument(**content) for content in structured_content_array]

            return structured_content_array
            
        except Exception as e:
            logger.error(f"Error extracting content from {doc_path}: {e}")
            raise Exception(f"Error extracting content from {doc_path}: {e}")

    def _encode_document(self, doc_path: str):
        logger.info(f"Encoding document {doc_path}")
        try:
            with open(doc_path, "rb") as doc_file:
                return base64.b64encode(doc_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding document {doc_path}: {e}")
            raise Exception(f"Error encoding document {doc_path}: {e}")




