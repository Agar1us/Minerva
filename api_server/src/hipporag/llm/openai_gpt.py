import asyncio
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import httpx
from filelock import FileLock
from openai import AsyncOpenAI

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


class SQLiteCacheManager:
    """
    Manages a thread-safe, asynchronous SQLite cache for LLM responses.

    This class handles key generation, database connections, and file-based locking
    to prevent race conditions in concurrent applications.
    """

    def __init__(self, db_path: str):
        """
        Initializes the cache manager and ensures the database and table exist.

        :param db_path: The file path for the SQLite database.
        """
        self.db_path = db_path
        self.lock_path = db_path + ".lock"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._setup_database()

    def _setup_database(self) -> None:
        """Creates the cache table if it doesn't already exist."""
        with FileLock(self.lock_path):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                    """
                )

    @staticmethod
    def _generate_cache_key(key_data: Dict[str, Any]) -> str:
        """
        Generates a deterministic SHA-256 hash for a given dictionary.

        :param key_data: The dictionary to hash.
        :return: A hex digest of the hash.
        """
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    async def get(self, key_data: Dict[str, Any]) -> Tuple[str, Dict] | None:
        """
        Retrieves an item from the cache asynchronously.

        :param key_data: The data used to generate the cache key.
        :return: A tuple of (message, metadata) if found, otherwise None.
        """
        key_hash = self._generate_cache_key(key_data)

        def _db_read():
            with FileLock(self.lock_path):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT message, metadata FROM cache WHERE key = ?", (key_hash,)
                    )
                    return cursor.fetchone()

        row = await asyncio.to_thread(_db_read)
        if row:
            message, metadata_str = row
            return message, json.loads(metadata_str)
        return None

    async def set(self, key_data: Dict[str, Any], message: str, metadata: Dict) -> None:
        """
        Saves an item to the cache asynchronously.

        :param key_data: The data used to generate the cache key.
        :param message: The LLM response message to cache.
        :param metadata: The metadata associated with the response.
        :return: None.
        """
        key_hash = self._generate_cache_key(key_data)
        metadata_str = json.dumps(metadata)

        def _db_write():
            with FileLock(self.lock_path):
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                        (key_hash, message, metadata_str),
                    )

        await asyncio.to_thread(_db_write)


class CacheOpenAI(BaseLLM):
    """
    An OpenAI LLM client with built-in caching and retry mechanisms.
    """

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        """
        Factory method to create an instance from a global configuration object.

        :param global_config: The configuration object for the experiment.
        :return: An instance of CacheOpenAI.
        """
        return cls(global_config=global_config)

    def __init__(
        self,
        global_config: BaseConfig,
        cache_filename: Optional[str] = None,
        high_throughput: bool = True,
    ) -> None:
        """
        Initializes the OpenAI client and the cache manager.

        :param global_config: The main configuration object.
        :param cache_filename: Optional custom name for the cache database file.
        :param high_throughput: If True, configures the HTTP client for higher concurrency.
        """
        super().__init__(global_config)
        self.cache_dir = os.path.join(self.global_config.save_dir, "llm_cache")

        if cache_filename is None:
            safe_llm_name = self.llm_name.replace("/", "_")
            cache_filename = f"{safe_llm_name}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self.cache = SQLiteCacheManager(self.cache_file_name)

        self._init_llm_config()

        client_kwargs = {
            "base_url": self.global_config.llm_base_url,
            "api_key": self.global_config.llm_api_key,
            "max_retries": self.llm_config.max_retries,
        }
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            timeout = httpx.Timeout(connect=60.0, read=300.0, write=60.0, pool=60.0)
            client_kwargs["http_client"] = httpx.AsyncClient(limits=limits, timeout=timeout)
        
        self.openai_client = AsyncOpenAI(**client_kwargs)

    def _init_llm_config(self) -> None:
        """
        Sets up the model-specific configuration from the global config.
        """
        config = self.global_config
        self.llm_config = LLMConfig.from_dict({
            "llm_name": config.llm_name,
            "llm_base_url": config.llm_base_url,
            "max_retries": config.max_retry_attempts,
            "generate_params": {
                "model": config.llm_name,
                "max_tokens": config.max_new_tokens,
                "n": config.num_gen_choices,
                "seed": config.seed,
                "temperature": config.temperature,
            },
        })
        logger.debug(f"Initialized {self.__class__.__name__}'s llm_config: {self.llm_config}")

    async def async_infer(
        self,
        messages: List[TextChatMessage],
        **kwargs,
    ) -> Tuple[str, Dict]:
        """
        Performs asynchronous inference, using the cache if possible.

        :param messages: A list of messages forming the chat history.
        :param kwargs: Additional parameters to override default generation settings.
        :return: A tuple containing:
                 - The response message (str).
                 - A metadata dictionary.
                 - A boolean indicating if the response was a cache hit.
        """
        params = deepcopy(self.llm_config.generate_params)
        params.update(kwargs)

        cache_key_data = {
            "messages": messages,
            "model": params.get("model"),
            "seed": params.get("seed"),
            "temperature": params.get("temperature"),
        }

        cached_result = await self.cache.get(cache_key_data)
        if cached_result:
            message, metadata = cached_result
            return message, metadata

        params["messages"] = messages
        logger.debug(f"Calling LLM API with:\n{params}")
        
        response = await self.openai_client.chat.completions.create(**params)
        
        response_message = response.choices[0].message.content
        if not isinstance(response_message, str):
            logger.warning("LLM response was not a string, coercing. Got: %s", type(response_message))
            response_message = str(response_message)

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        await self.cache.set(cache_key_data, response_message, metadata)

        return response_message, metadata
