import os
import fitz
from document_processing.models.document import Document
from typing import List


import logging
logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads PDF documents using PyMuPDF (fitz) and converts each page into a Document object.

    This class provides functionality to load PDF files, extract text and metadata from each page,
    and store the results in a structured format for further processing.
    """

    @staticmethod
    def load(file_path: str) -> List[Document]:
        """
        Loads a PDF file and returns a list of Document objects, one for each page.

        Args:
            - file_path (str): The path to the PDF file.

        Returns:
            - List[Document]: A list of Document objects, each representing a page in the PDF.

        Raises:
            - FileNotFoundError: If the file does not exist.
            - ValueError:
                - If the file is not a PDF.
                - If the file is corrupted.
            - RuntimeError: If an error occurs while loading the PDF file.
        """

        try:
            logger.info(f"Loading PDF file: {file_path}")

            # Check if the file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            
            # Check if the file is a PDF
            if not file_path.endswith(".pdf"):
                logger.error(f"Unsupported file format: {file_path}")
                raise ValueError("Only PDF files are supported")

            # Load the PDF document
            with fitz.open(file_path) as doc:
                pages = []
                # Iterate through each page and extract content and metadata
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    page_content = page.get_text()
                    metadata = doc.metadata.copy()
                    metadata.update({
                        "source": os.path.abspath(file_path),
                        "page": page_num
                    })
                    # Ensure metadata values are strings
                    metadata = {k: v if v is not None else "" for k,v in metadata.items()}
                    pages.append(Document(page_content, metadata))

            logger.info(f"Successfully loaded {len(pages)} pages from {file_path}")
            return pages
        except fitz.FileDataError as e:
            logger.exception(f"Corrupted or invalid file: {file_path}")
            raise ValueError(f"The file '{file_path}' is not a valid file:\n{e}")
        except Exception as e:
            logger.exception(f"Failed to load file: {file_path}")
            raise RuntimeError(f"An error occurred while loading the file '{file_path}':\n{e}")
