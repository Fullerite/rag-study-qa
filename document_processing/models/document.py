class Document:
    """
    Represents a document with its content and associated metadata.
    """

    def __init__(self, page_content: str, metadata: dict):
        """
        Initializes a Document object.

        Args:
            - page_content: The text content of the document.
            - metadata: A dictionary containing metadata about the document.
        """

        self.page_content = page_content
        self.metadata = metadata

    def __str__(self):
        return (
            f"Document(\n"
            f"  page_content={self.page_content},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )

    def __repr__(self):
        return (
            f"Document(\n"
            f"  page_content={repr(self.page_content)},\n"
            f"  metadata={repr(self.metadata)})\n"
            f")"
        )
