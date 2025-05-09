import re
from document_processing.models.document import Document
from typing import List


import logging
logger = logging.getLogger(__name__)


class DocumentTaskSplitter:
    """
    Splits a document into tasks based on specified patterns for questions and answers.

    This class is designed to process documents by identifying tasks (e.g., questions) and their
    corresponding answers using regular expressions. It also supports adding start and end indices
    to the metadata for each task.
    """

    def __init__(
        self,
        task_pattern: str,
        answer_pattern: str
    ):
        """
        Initializes the DocumentTaskSplitter with task and answer patterns.

        Args:
            - task_pattern (str): The regular expression pattern for identifying task beginnings.
            - answer_pattern (str): The regular expression pattern for identifying answers within tasks.
        
        Raises:
            - ValueError: If the task or answer pattern is invalid.
        """

        self._task_pattern = task_pattern
        self._answer_pattern = answer_pattern

        try:
            re.compile(self._task_pattern)
            re.compile(self._answer_pattern)
        except re.error as e:
            logger.exception(f"Regex compilation error: {e.pattern}")
            raise ValueError(f"Invalid regex pattern: {e.pattern}. Error compiling at position {e.pos}")


    def __str__(self):
        return (
            f"DocumentTaskSplitter(\n"
            f"  task_pattern={self._task_pattern},\n"
            f"  answer_pattern={self._answer_pattern}\n"
            f")"
        )

    def __repr__(self):
        return (
            f"DocumentTaskSplitter(\n"
            f"  task_pattern={self._task_pattern},\n"
            f"  answer_pattern={self._answer_pattern}\n"
            f")"
        )


    def split_text(self, text: str) -> List[dict]:
        """
        Splits a text into individual tasks based on the defined patterns.

        Args:
            - text (str): The text to split.

        Returns:
            - List[Dict]: A list of dictionaries, each containing the 'question_statement' and 'question_full'.
                - "question_statement" (str): The question statement.
                - "question_full" (str): The full content of the question, including the answer options.

        Raises:
            - RuntimeError: If an error occurs while splitting the text.
        """

        try:        
            logger.info("Splitting text into tasks")

            if not text:
                logger.warning("Input text is empty")
                return []

            # Find all matches for tasks in the text
            questions = []
            matches = re.finditer(self._task_pattern, text, re.DOTALL)

            # Split the tasks text based on the answer pattern
            for match in matches:
                # Extract the whole question text
                question_full = match.group(1).strip()

                parts = re.split(self._answer_pattern, question_full, flags=re.MULTILINE)
                if parts:
                    # Extract the question statement
                    question_statement = parts[0].strip()
                    questions.append(
                        {
                            "question_statement": question_statement,
                            "question_full": question_full
                        }
                    )
                else:
                    logger.warning(f"Could not separate question statement from answers for question:\n\"\"\"\n{question_full}\n\"\"\"")

            logger.info(f"Found {len(questions)} questions in text")
            return questions
        except Exception as e:
            logger.exception(f"Error splitting text: {text}")
            raise RuntimeError(f"An error occurred while splitting text:\n{e}")


    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller documents based on identified tasks.

        Args:
            - documents (List[Document]): The list of Document objects to split.

        Returns:
            - List[Document]: A list of Document objects, each representing an individual task.
        
        Raises:
            - ValueError: If the input documents list is empty.
            - RuntimeError: If an error occurs while splitting the documents.
        """

        try:
            logger.info(f"Splitting {len(documents)} documents into tasks")

            if not documents:
                logger.error("Input documents are empty")
                raise ValueError("The input documents cannot be empty.")

            # Process each document
            split_documents = []
            for document in documents:
                # Split the document into questions
                chunks = self.split_text(document.page_content)
                start_index = 0
                for chunk in chunks:
                    question_statement, question_full = chunk["question_statement"], chunk["question_full"]
                    metadata = document.metadata.copy()

                    # Add start and end indices
                    start_index = document.page_content.find(question_statement)
                    end_index = start_index + len(question_full)
                    metadata["start_index"] = start_index
                    metadata["end_index"] = end_index

                    # Create a new document for each task
                    new_document = Document(page_content=question_statement, metadata=metadata)
                    split_documents.append(new_document)

            logger.info(f"Created {len(split_documents)} task documents")
            return split_documents
        except Exception as e:
            logger.exception(f"Error splitting documents: {documents}")
            raise RuntimeError(f"An error occurred while splitting documents:\n{e}")
