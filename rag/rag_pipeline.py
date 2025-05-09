import os
import uuid
import chromadb

from rag.generation.generation_model import TransformersGenerationModel
from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.embedding.embedding_function import SentenceTransformerEmbeddingFunction
from document_processing.document_processor import DocumentProcessor

from typing import Optional, Union, Tuple, Generator
from exceptions.custom_exceptions import RAGInitializationError, CorpusCreationError, QueryProcessingError


import logging
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    A class for implementing a Retrieval-Augmented Generation (RAG) pipeline.

    This class integrates a generation model, an embedding model, and a vector database (ChromaDB)
    to create a knowledge corpus and answer user queries based on retrieved context.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformersEmbeddingModel,
        generation_model: TransformersGenerationModel
    ):
        """
        Initializes the RAGPipeline with a generation model and an embedding model.

        Args:
            - generation_model (TransformersGenerationModel): The model used for text generation.
            - embedding_model (SentenceTransformersEmbeddingModel): The model used for generating embeddings.
        
        Raises:
            - RAGInitializationError: If an error occurs while initializing the RAG pipeline.
        """

        try:
            logger.info("Initializing RAG pipeline")
            self._embedding_model = embedding_model
            self._generation_model = generation_model
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name="test_collection",
                embedding_function=SentenceTransformerEmbeddingFunction(self._embedding_model),
                metadata={
                    "hnsw:space": "cosine"
                }
            )
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.exception(
                f"Failed to initialize RAG pipeline\n"
                f"Embedding Model: {repr(self._embedding_model)}\n"
                f"Generation Model: {repr(self._generation_model)}"
            )
            raise RAGInitializationError(
                f"An error occurred while initializing the RAG pipeline:\n{e}",
                embedding_model_info=repr(self._embedding_model),
                generation_model_info=repr(self._generation_model)
            )


    def create_knowledge_corpus(
        self,
        data_dir: str,
        task_pattern: str,
        answer_pattern:str,
    ) -> None:
        """
        Creates a knowledge corpus by processing PDF files in the specified directory.

        Args:
            - data_dir (str): The directory containing the PDF files.
            - task_pattern (str): The regex pattern used to identify tasks within the documents.
            - answer_pattern (str): The regex pattern used to identify answers within the documents.
        """

        try:
            logger.info(f"Creating knowledge corpus from directory: {repr(data_dir)}")

            # Process each PDF file in the directory
            corpus_passages = []
            corpus_metadatas = []
            for filename in os.listdir(data_dir):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(data_dir, filename)
                    passages, metadatas = DocumentProcessor.process_file(
                        file_path=file_path,
                        task_pattern=task_pattern,
                        answer_pattern=answer_pattern
                    )
                    corpus_passages.extend(passages)
                    corpus_metadatas.extend(metadatas)

            # Add the processed passages and metadata to the ChromaDB collection
            if corpus_passages:
                self.collection.add(
                    documents=corpus_passages,
                    metadatas=corpus_metadatas,
                    ids=[str(uuid.uuid4()) for _ in range(len(corpus_passages))]
                )
                logger.info(f"Added {len(corpus_passages)} passages to knowledge corpus")
            else:
                logger.warning("No valid passages were found. Knowledge corpus is empty")
        except Exception as e:
            logger.exception(
                f"Failed to create knowledge corpus\n"
                f"Data Directory: {repr(data_dir)}\n"
                f"Task Pattern: {repr(task_pattern)}\n"
                f"Answer Pattern: {repr(answer_pattern)}\n"
                f"ChromaDB client: {repr(self.client)}"
                f"ChromaDB collection: {repr(self.collection)}"
            )
            raise CorpusCreationError(
                f"An error occurred while creating the knowledge corpus:\n{e}",
                data_dir=data_dir,
                task_pattern=task_pattern,
                answer_pattern=answer_pattern,
                chromadb_client_info=repr(self.client),
                chromadb_collection_info=repr(self.collection)
            )

    
    def query(
        self,
        user_query: str,
        system_prompt: str = "",
        context_window_backward: Optional[int] = None,
        context_window_forward: Optional[int] = None,
        stream_output: bool = False
    ) -> Tuple[Union[Generator[str, None, None], str], str]:
        """
        Answers a user query by retrieving relevant context from the knowledge corpus and generating a response.

        Args:
            - user_query (str): The user's query.
            - system_prompt (str): The system prompt to guide the generation model.
                                    Defaults to "".
            - context_window_backward (Optional[int]): The number of characters to include before the retrieved context.
                                                        Defaults to None.
            - context_window_forward (Optional[int]): The number of characters to include after the retrieved context.
                                                        Defaults to None.
            - stream_output (bool): Whether to stream the output in real-time.
                                    Defaults to False.

        Returns:
            - Union[str, Generator[str, None, None]]:
                - stream_output=False: The complete generated text string.
                - stream_output=True: An generator that yields chunks of generated text as they are produced.
            - str: The context fetched from the knowledge corpus and used by the generation model.
        """

        try:
            logger.info(f"Processing query: {repr(user_query)}")

            # Extract test and question numbers from the query
            metadata_filter = {}
            user_query_test_number = DocumentProcessor.extract_test_number(user_query)
            user_query_question_number = DocumentProcessor.extract_question_number(user_query)
            if user_query_test_number is not None:
                metadata_filter["test"] = user_query_test_number
            if user_query_question_number is not None:
                metadata_filter["question"] = user_query_question_number
            if len(metadata_filter) > 1:
                metadata_filter = {"$and": [{"test": user_query_test_number}, {"question": user_query_question_number}]}

            # Query the ChromaDB collection for the most relevant passage
            result = self.collection.query(
                query_texts=[user_query],
                where=metadata_filter if metadata_filter else None,
                n_results=1
            )

            # Filter cases with non-valid test or question numbers
            if not result["documents"][0]:
                logger.warning(
                    f"No relevant documents found for query {repr(user_query)} "
                    f"with filter {metadata_filter}"
                )
                return "No relevant information found in the files. Double check your question and test numbers or the data folder.", "No context"

            # Retrieve metadata and context for the best-matching passage
            best_fit_metadata = result["metadatas"][0][0]
            file_path = best_fit_metadata["source"]
            page = best_fit_metadata["page"]
            start_index = best_fit_metadata["start_index"]
            end_index = best_fit_metadata["end_index"]

            # Set default context window values if not provided
            context_window_backward = (
                context_window_backward if context_window_backward is not None else 0
            )  # from the start
            context_window_forward = (
                context_window_forward if context_window_forward is not None else end_index - start_index
            )  # to the end of the question content
            
            # Retrieve the passage with the specified context window
            passage_context = DocumentProcessor.retrieve_task(
                file_path=file_path,
                page=page,
                start_index=start_index,
                context_window_backward=context_window_backward,
                context_window_forward=context_window_forward
            )

            logger.info(f"Retrieved context:\n\"\"\"\n{passage_context}\n\"\"\"")
            logger.info("Generating answer using retrieved context")

            # Generate the answer using the generation model
            query_answer = self._generation_model.generate(
                query=f"{user_query}\n\n{passage_context}",
                system_prompt=system_prompt,
                stream_output=stream_output
            )

            return query_answer, passage_context
        except Exception as e:
            logger.exception(
                f"Failed to process query\n"
                f"Query: '{user_query}'\n"
                f"System prompt: '{system_prompt}'\n"
                f"Embedding Model: {repr(self._embedding_model)}\n"
                f"Generation Model: {repr(self._generation_model)}\n"
                f"ChromaDB client: {repr(self.client)}"
                f"ChromaDB collection: {repr(self.collection)}"
            )
            raise QueryProcessingError(
                f"An error occurred while processing the query:\n{e}",
                query=user_query,
                embedding_model_info=repr(self._embedding_model),
                generation_model_info=repr(self._generation_model),
                chromadb_client_info=repr(self.client),
                chromadb_collection_info=repr(self.collection)
            )
