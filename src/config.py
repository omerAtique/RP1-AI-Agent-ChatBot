from dotenv import load_dotenv
import os
import uuid
import logging
from rich.logging import RichHandler
import json
from pathlib import Path

load_dotenv()

class MistralConfig:
    api_key: str = os.getenv("MISTRAL_API_KEY")
    model: str = "mistral-large-latest"
    
class ETLConfig:
    docs_folder_path: str = os.getenv("DOCUMENTS_PATH")

class LangChainConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_LLM_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    model_provider: str = "openai"

class LCQdrantConfig:
    embedding_model: str = "text-embedding-3-small"
    model_provider: str = "openai"
    embedding_size: int = 1536
    distance: str = "Cosine"
    
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        self.input_file_path = os.getenv("INPUT_FILE_PATH")
        self.collection_name = self._get_or_create_collection_name()
    
    def _get_or_create_collection_name(self) -> str:
        try:
            input_file_path = Path(self.input_file_path)
            
            if input_file_path.exists():
                try:
                    with open(input_file_path, 'r') as f:
                        data = json.load(f)
                        collection_name = data.get("collection_name", "")
                        if collection_name and collection_name.strip():
                            return collection_name
                except (json.JSONDecodeError, IOError):
                    # If file exists but can't be read, fall through to create new name
                    pass
            
            # Generate new collection name
            new_collection_name = f"Vector_Store_{str(uuid.uuid4())[:8]}"

            # update input.json with the new collection name
            data["collection_name"] = new_collection_name
            with open(input_file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return new_collection_name
        except Exception as e:
            raise Exception(f"Error in _get_or_create_collection_name: {e}")

class AppConfig:
    environment: str = os.getenv("ENVIRONMENT", "development")
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    timeout_seconds: int = int(os.getenv("TIMEOUT_SECONDS", "120"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

class LoggingConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")
    console_enabled: bool = os.getenv("LOG_CONSOLE_ENABLED", "True").lower() == "true"
    file_enabled: bool = os.getenv("LOG_FILE_ENABLED", "True").lower() == "true"
   
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class Config:
    def __init__(self, validate_on_init: bool = False):
        self.mistral = MistralConfig()
        self.etl = ETLConfig()
        self.langchain = LangChainConfig()
        self.lc_qdrant = LCQdrantConfig()
        self.app = AppConfig()
        self.logging = LoggingConfig()

        self._setup_logging()

        if validate_on_init:
            self._validate_required_configs()

    def _validate_required_configs(self):
        required_configs = [
            (self.mistral.api_key, "MISTRAL_API_KEY"),
            (self.etl.docs_folder_path, "DOCUMENTS_PATH"),
            (self.langchain.openai_api_key, "OPENAI_API_KEY"),
            (self.lc_qdrant.qdrant_url, "QDRANT_URL"),
            (self.lc_qdrant.qdrant_api_key, "QDRANT_API_KEY"),
            (self.lc_qdrant.qdrant_collection_name, "QDRANT_COLLECTION_NAME"),
        ]

        missing_configs = [config for config in required_configs if not config[0]]

        if missing_configs:
            raise ValueError(f"Missing required configurations: {', '.join([config[1] for config in missing_configs])}")

    def _setup_logging(self):
        logging_config = self.logging

        logger = logging.getLogger()
        logger.setLevel(getattr(logging, logging_config.log_level.upper()))
        
        logger.handlers.clear()

        if logging_config.console_enabled:
            console_handler = RichHandler(
                show_time=True,
                show_path=True,
                enable_link_path=False,
                rich_tracebacks=True
            )
            console_formatter = logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        if logging_config.file_enabled:
            os.makedirs(os.path.dirname(logging_config.log_file), exist_ok=True)
            file_handler = logging.FileHandler(logging_config.log_file)
            file_formatter = logging.Formatter(logging_config.log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

config = Config()
logger = logging.getLogger(__name__)


