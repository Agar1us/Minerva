from typing import List, Optional

import numpy as np
from openai import OpenAI, APIError
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    An embedding model that uses an OpenAI-compatible API to generate embeddings.
    This can be used with OpenAI's official models or any other service that
    exposes a compatible `/v1/embeddings` endpoint.
    """

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
    ):
        """
        Initializes the OpenAI API client and model configuration.

        :param global_config: An optional global configuration object for default values.
        """
        super().__init__(global_config=global_config)
        self.embedding_model_name = global_config.embedding_name

        logger.info(f"Initializing OpenAI client for model {self.embedding_model_name}")
        try:
            self.client = OpenAI(
                api_key=self.global_config.embedding_api_key,
                base_url=self.global_config.embedding_base_url
                )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        self._setup_embedding_config()

    def _setup_embedding_config(self):
        """Initializes the EmbeddingConfig with parameters for encoding."""
        self.embedding_config = {
            "normalize_embeddings": self.global_config.embedding_return_as_normalized,
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
            },
        }
        logger.debug(f"Initialized {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Prepares texts for the OpenAI API by replacing newlines and handling empty strings.
        
        :param texts: A list of raw strings.
        :return: A list of processed strings.
        """
        return [
            t.replace("\n", " ").strip() or " " for t in texts
        ]

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a single batch of texts into NumPy embeddings using the OpenAI API.

        :param texts: A list of processed strings to encode.
        :return: A NumPy array containing the embeddings.
        :raises APIError: If the API call fails.
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_model_name
        )
        embeddings = np.array([v.embedding for v in response.data])
            
        return embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encodes a list of texts in batches and returns a single NumPy array of embeddings.

        :param texts: A list of strings to encode.
        :param kwargs: Additional parameters (e.g., 'batch_size').
        :return: A NumPy array of shape (num_texts, embedding_dim).
        """
        params = self.embedding_config["encode_params"].copy()
        params.update(kwargs)
        batch_size = params["batch_size"]

        processed_texts = self._preprocess_texts(texts)
        all_embeddings = []
        
        pbar = tqdm(
            total=len(processed_texts),
            desc="Batch Encoding (OpenAI API)",
            disable=len(processed_texts) <= batch_size
        )

        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            try:
                batch_embeddings = self.encode(batch)
                all_embeddings.append(batch_embeddings)
            except APIError as e:
                logger.error(f"OpenAI API error on batch starting at index {i}: {e}")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred on batch starting at index {i}: {e}")
                raise
            pbar.update(len(batch))
        
        pbar.close()

        if not all_embeddings:
            return np.array([])
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        return self.postprocess_embeddings(embeddings)

    def postprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Optionally normalizes embeddings. The input is already a NumPy array.
        
        :param embeddings: A NumPy array of embeddings.
        :return: A processed NumPy array of embeddings.
        """
        if self.embedding_config.get("normalize_embeddings", False):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            embeddings = embeddings / norms
            
        return embeddings