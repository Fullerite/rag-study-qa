from chromadb import EmbeddingFunction
from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from typing import List, Union
from numpy import ndarray


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    A ChromaDB-compatible embedding function that uses a SentenceTransformers model to generate embeddings.

    This class wraps a SentenceTransformers model and provides a callable interface for generating embeddings
    from text inputs.
    """

    def __init__(self, model: SentenceTransformersEmbeddingModel):
        """
        Initializes the SentenceTransformerEmbeddingFunction with a SentenceTransformers model.

        Args:
            - model (SentenceTransformersEmbeddingModel): The SentenceTransformers model to use for generating embeddings.
        """

        self._model = model


    def __str__(self):
        return (
            f"SentenceTransformerEmbeddingFunction(\n"
            f"  model={self._model}\n"
            f")"
        )


    def __repr__(self):
        return (
            f"SentenceTransformerEmbeddingFunction(\n"
            f"  model={repr(self._model)}\n"
            f")"
        )


    def __call__(self, sentences: Union[str, List[str]]) -> ndarray:
        """
        Generates embeddings for the input sentences using the SentenceTransformers model.

        Args:
            - sentences (str | List[str]): A single sentence or a list of sentences to embed.

        Returns:
            - ndarray: A numpy array containing the embeddings for the input sentences.
                      The shape is (num_sentences, embedding_dimension).
        """

        # Generate embeddings using the SentenceTransformers model
        embeddings = self._model.encode(sentences)

        return embeddings
