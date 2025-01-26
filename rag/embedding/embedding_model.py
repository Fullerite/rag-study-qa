import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
from numpy import ndarray
from exceptions.custom_exceptions import ModelLoadingError, EncodingError


import logging
logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingModel:
    """
    A wrapper class for a SentenceTransformers model that generates embeddings for text inputs.

    This class handles model initialization, device management (CPU/GPU), and provides a method
    to encode sentences into embeddings.
    """

    def __init__(self, model_name: str):
        """
        Initializes the SentenceTransformersEmbeddingModel with the specified model and configuration.
        Uses half-precision (FP16).

        Args:
            - model_name (str): The name or path of the SentenceTransformers model to load.
        
        Raises:
            - ModelLoadingError: If an error occurs while loading the SentenceTransformers model.
        """

        logger.info(f"Initializing embedding model: {model_name}")
        try:
            self.model_name = model_name
            self._model = SentenceTransformer(
                model_name_or_path=model_name,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                model_kwargs={"torch_dtype": torch.float16},
                trust_remote_code=True
            )
            self.device = self._model.device
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.exception(
                f"Failed to load the model\n"
                f"Model: {self.model_name}\n"
                f"Quantization Config: None"
            )
            raise ModelLoadingError(
                f"An error occurred while loading the model {model_name}:\n{e}",
                model_name=self.model_name,
            )


    def __str__(self):
        return (
            f"SentenceTransformersEmbeddingModel(\n"
            f"  model_name={self.model_name},\n"
            f"  device={self.device}\n"
            f")"
        )


    def __repr__(self):
        return (
            f"SentenceTransformersEmbeddingModel(\n"
            f"  model_name={self.model_name},\n"
            f"  model={repr(self._model)},\n"
            f"  device={self.device}\n"
            f")"
        )


    def encode(self, sentences: Union[str, List[str]]) -> ndarray:
        """
        Generates embeddings for the input sentences using the SentenceTransformers model.

        Args:
            - sentences (str | List[str]): A single sentence or a list of sentences to embed.

        Returns:
            - ndarray: A numpy array containing the embeddings for the input sentences.

        Raises:
            - EncodingError: If an error occurs during encoding.
        """

        try:
            logger.info(f"embedding {len(sentences) if isinstance(sentences, list) else 1} sentence(s)")

            # Generate embeddings using the SentenceTransformers model
            encoding_kwargs = {
                "normalize_embeddings": True,
                "convert_to_numpy": True
            }
            embeddings = self._model.encode(
                sentences,
                **encoding_kwargs
            )

            return embeddings
        except Exception as e:
            logger.exception(
                f"Failed to encode sentence(s) '{sentences}' using:\n\n{repr(self)}\n"
                f"Sentence(s): {sentences}\n"
                f"Model: {repr(self)}"
            )
            raise EncodingError(
                f"An error occured while encoding sentences:\n{e}",
                sentences=sentences,
                encoding_kwargs=encoding_kwargs,
                model_info=repr(self)
            )
