import json
from dataclasses import dataclass, field, asdict
from typing import (
    Optional,
    Any, 
    Dict,
    List
)
import multiprocessing

import numpy as np

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..llm.base import LLMConfig

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:
        ignored_prefixes = ("_ipython_", "_repr_")
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
        if key in self._data:
            return self._data[key]
        
        logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


    def __setattr__(self, key: str, value: Any) -> None:
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key lookup."""
        if key in self._data:
            return self._data[key]
        logger.error(f"'{key}' not found in configuration.")
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style key assignment."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Allow dict-style key deletion."""
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{key}' not found in configuration.")
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:
        """Allow usage of 'in' to check for keys."""
        return key in self._data
    
    
    def batch_upsert(self, updates: Dict[str, Any]) -> None:
        """Update existing attributes or add new ones from the given dictionary."""
        self._data.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """Export the configuration as a JSON-serializable dictionary."""
        return self._data

    def to_json(self) -> str:
        """Export the configuration as a JSON string."""
        return json.dumps(self._data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> LLMConfig:
        """Create an LLMConfig instance from a dictionary."""
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> LLMConfig:
        """Create an LLMConfig instance from a JSON string."""
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the configuration."""
        return json.dumps(self._data, indent=4)

    
class BaseEmbeddingModel:
    """
    Abstract base class for all embedding models.
    Defines the common interface for encoding text and calculating scores.
    """
    global_config: BaseConfig
    embedding_model_name: str
    embedding_config: EmbeddingConfig
    
    embedding_dim: int
    
    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None: 
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else: self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")
        
        
        self.embedding_model_name = self.global_config.embedding_model_name

        logger.debug(f"Init {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        """
        Encodes a batch of texts into embeddings.
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError
    
    
    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        """
        Calculates the dot product similarity scores between a query vector
        and a matrix of document vectors.

        :param query_vec: A 1D NumPy array for the query embedding.
        :param doc_vecs: A 2D NumPy array where each row is a document embedding.
        :return: A 1D NumPy array of similarity scores.
        """
        return np.dot(query_vec, doc_vecs.T)
    


class EmbeddingCache:
    _manager = None
    _cache = None
    _lock = None

    @classmethod
    def _ensure_initialized(cls):
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
            cls._cache = cls._manager.dict()
            cls._lock = cls._manager.Lock()

    @classmethod
    def get(cls, content):
        cls._ensure_initialized()
        return cls._cache.get(content)

    @classmethod
    def set(cls, content, embedding):
        cls._ensure_initialized()
        with cls._lock:  # Межпроцессная блокировка
            cls._cache[content] = embedding

    @classmethod
    def contains(cls, content):
        cls._ensure_initialized()
        return content in cls._cache

    @classmethod
    def clear(cls):
        cls._ensure_initialized()
        with cls._lock:
            cls._cache.clear()