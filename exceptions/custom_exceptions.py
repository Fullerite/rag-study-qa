from transformers import BitsAndBytesConfig
from typing import List, Optional, Union


class ModelLoadingError(Exception):
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None
    ):
        self.message = message
        self.model_name = model_name
        self.quantization_config = quantization_config
        super().__init__(self.message)
        

class EncodingError(Exception):
    def __init__(
        self,
        message: str,
        sentences: Optional[Union[str, List[str]]] = None,
        encoding_kwargs: Optional[dict] = None,
        model_info: Optional[str] = None
    ):
        self.message = message
        self.sentences = sentences
        self.encoding_kwargs = encoding_kwargs
        self.model_info = model_info
        super().__init__(self.message)


class TemplateTokenizationError(Exception):
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_info: Optional[str] = None
    ):
        self.query = query
        self.system_prompt = system_prompt
        self.model_info = model_info
        super().__init__(self.message)


class GenerationError(Exception):
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream_output: Optional[bool] = None,
        generation_kwargs: Optional[dict] = None,
        model_info: Optional[str] = None
    ):
        self.message = message
        self.query = query
        self.system_prompt = system_prompt
        self.stream_output = stream_output
        self.generation_kwargs = generation_kwargs
        self.model_info = model_info
        super().__init__(self.message)


class RAGInitializationError(Exception):
    def __init__(
        self,
        message: str,
        embedding_model_info: Optional[str] = None,
        generation_model_info: Optional[str] = None
    ):
        self.message = message
        self.generation_model_info = generation_model_info
        self.embedding_model_info = embedding_model_info
        super().__init__(self.message)


class CorpusCreationError(Exception):
    def __init__(
        self,
        message: str,
        data_dir: Optional[str] = None,
        task_pattern: Optional[str] = None,
        answer_pattern: Optional[str] = None,
        chromadb_client_info: Optional[str] = None,
        chromadb_collection_info: Optional[str] = None
    ):
        self.message = message
        self.data_dir = data_dir
        self.task_pattern = task_pattern
        self.answer_pattern = answer_pattern
        self.chromadb_client_info = chromadb_client_info
        self.chromadb_collection_info = chromadb_collection_info
        super().__init__(self.message)


class QueryProcessingError(Exception):
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        embedding_model_info: Optional[str] = None,
        generation_model_info: Optional[str] = None,
        chromadb_client_info: Optional[str] = None,
        chromadb_collection_info: Optional[str] = None
    ):
        self.message = message
        self.query = query
        self.embedding_model_info = embedding_model_info
        self.generation_model_info = generation_model_info
        self.chromadb_client_info = chromadb_client_info
        self.chromadb_collection_info = chromadb_collection_info
        super().__init__(self.message)
