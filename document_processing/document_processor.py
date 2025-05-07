import re
from document_processing.document_loader import PDFLoader
from document_processing.document_splitter import DocumentTaskSplitter
from typing import List, Optional, Tuple


import logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Provides utility methods for processing documents, including extracting
    information, splitting documents into tasks, and retrieving specific
    text passages.
    """

    @staticmethod
    def extract_test_number(text: str) -> Optional[int]:
        """
        Extracts the test number from a given text string.

        Args:
            - text (str): The text to extract the test number from.

        Returns:
            - Optional[int]: The test number as an integer, or None if no test number is found.
        """

        match = re.search(r"(test|variant)\s*(\d+)", text, re.IGNORECASE)
        test_number = int(match.group(2)) if match else None
        return test_number
    

    @staticmethod
    def extract_question_number(text: str) -> Optional[int]:
        """
        Extracts the question number from a given text string.

        Args:
            - text (str): The text to extract the question number from.

        Returns:
            - Optional[int]: The question number as an integer, or None if no question number is found.
        """

        match = re.search(r"(question|q|problem|task)\s*(\d+)", text, re.IGNORECASE)
        question_number = int(match.group(2)) if match else None
        return question_number


    @staticmethod
    def process_file(
        file_path: str,
        task_pattern: str,
        answer_pattern:str,
    ) -> Tuple[List[str], List[dict]]:
        """
        Processes a PDF file, extracting text passages and associated metadata.

        Args:
            - file_path (str): The path to the PDF file.
            - task_pattern (str): The regular expression pattern used to identify tasks within the document.
            - answer_pattern (str): The regular expression pattern used to identify answers within the document.

        Returns:
            - Tuple[List[str], List[dict]]: A tuple containing two lists:
                - A list of text passages.
                - A list of metadata dictionaries corresponding to each passage.
        
        Raises:
            - ValueError: If the file is empty.
            - RuntimeError: If an error occurs while processing the file.
        """

        try:
            logger.info(f"Processing file: {file_path}")

            # Load the document and extract the test number from the first page
            document = PDFLoader.load(file_path)
            if not document:
                logger.error(f"The document '{file_path}' is empty")
                raise ValueError(f"The document '{file_path}' is empty.")
            test_number = DocumentProcessor.extract_test_number(document[0].page_content)

            # Initialize the text splitter with the specified patterns and settings
            text_splitter = DocumentTaskSplitter(
                task_pattern=task_pattern,
                answer_pattern=answer_pattern
            )
            
            # Split the document into chunks and process each chunk
            passages = []
            metadatas = []
            for chunk in text_splitter.split_documents(document):
                question_number = DocumentProcessor.extract_question_number(chunk.page_content)
                chunk.metadata["test"] = test_number
                chunk.metadata["question"] = question_number
                passages.append(f"Test {test_number}. {chunk.page_content}")
                metadatas.append(chunk.metadata)

            logger.info(f"Processed {len(passages)} passages from {file_path}")
            return passages, metadatas
        except Exception as e:
            logger.error(f"Error processing file: {file_path}")
            raise RuntimeError(f"An error occurred while processing the file '{file_path}':\n{e}")


    @staticmethod
    def retrieve_task(
        file_path: str,
        page: int,
        start_index: int,
        context_window_backward: int = 0,
        context_window_forward: int = 200
    ) -> str:
        """
        Retrieves a specific task from a document with the surrounding context.

        Args:
            - file_path (str): The path to the PDF file.
            - page (int): The page number to retrieve the task from.
            - start_index (int): The starting index of the task within the page text.
            - context_window_backward (int): The number of characters to include before the start_index as context. Defaults to 0.
            - context_window_forward (int): The number of characters to include after the start_index as context. Defaults to 200.

        Returns:
            - str: The text passage retrieved from the document, including the specified context.
        
        Raises:
            - ValueError:
                - If the page or start index is out of bounds.
                - If the context window parameters are negative.
            - RuntimeError: 
                - If an unexpected error occurs while retrieving the task.
        """

        try:
            logger.info("Retrieving the passage context from the source file")

            # Load the document and extract the page text
            document = PDFLoader.load(file_path)
            if page < 0 or page >= len(document):
                logger.error(f"Page {page} is out of bounds")
                raise ValueError(f"Page number is out of bounds.")
            page_text = document[page].page_content
            if start_index < 0 or start_index >= len(page_text):
                logger.error(f"Start index {start_index} is out of bounds")
                raise ValueError(f"Start index is out of bounds.")

            if context_window_backward < 0 or context_window_forward < 0:
                logger.error(f"Context window parameters ({context_window_backward=}, {context_window_forward=}) must be non-negative")
                raise ValueError("Context window parameters must be non-negative.")

            # Calculate start and end indices for context window
            start = max(0, start_index - context_window_backward)
            end = min(len(page_text), start_index + context_window_forward)

            return page_text[start:end]
        except Exception as e:
            logger.exception(f"Failed to retrieve task from file: {file_path}")
            raise RuntimeError(f"An error occurred while retrieving the task:\n{e}")
