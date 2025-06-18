import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#                              LLMConfig wrapper                              #
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """
    Flexible key–value store that behaves both as an object and a dict.

    All attributes are stored in the private ``_data`` field.
    Direct attribute access (`conf.foo`) is therefore equivalent to
    dict-style access (`conf['foo']`).
    """

    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:
        """
        Fetches a value using attribute syntax.

        :param key: Name of the attribute.
        :return:    Stored value if the key exists; raises ``AttributeError`` otherwise.
        """
        ignored_prefixes = ("_ipython_", "_repr_")
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        if key in self._data:
            return self._data[key]

        logger.error("'%s' object has no attribute '%s'", self.__class__.__name__, key)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Assigns a value using attribute syntax.

        :param key:   Name to be set.
        :param value: Value to assign.
        """
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        """
        Deletes an attribute.

        :param key: Name of the attribute to delete.
        """
        if key in self._data:
            del self._data[key]
        else:
            logger.error("'%s' object has no attribute '%s'", self.__class__.__name__, key)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    # ------------------------------ dict-like ----------------------------- #
    def __getitem__(self, key: str) -> Any:
        """
        Dict-style read access.

        :param key: Key to look up.
        :return:    Corresponding value.
        :raises KeyError: If the key is absent.
        """
        if key in self._data:
            return self._data[key]
        logger.error("'%s' not found in configuration.", key)
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Dict-style assignment.

        :param key:   Key to create or overwrite.
        :param value: Value to store.
        """
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """
        Dict-style deletion.

        :param key: Key to remove.
        :raises KeyError: If the key is absent.
        """
        if key in self._data:
            del self._data[key]
        else:
            logger.error("'%s' not found in configuration.", key)
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:
        """
        Enables the ``in`` operator.

        :param key: Key to check.
        :return:    ``True`` if key exists, else ``False``.
        """
        return key in self._data

    # ------------------------------ helpers ------------------------------ #
    def batch_upsert(self, updates: Dict[str, Any]) -> None:
        """
        Adds or replaces multiple entries at once.

        :param updates: Mapping of keys and values to merge into the config.
        :return:        ``None``.
        """
        self._data.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the configuration to a plain dictionary.

        :return: A shallow copy of the internal data.
        """
        return self._data

    def to_json(self) -> str:
        """
        Serialises the configuration to a JSON string.

        :return: JSON-encoded string.
        """
        return json.dumps(self._data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """
        Creates an instance from an existing dictionary.

        :param config_dict: Source data.
        :return:            Initialised ``LLMConfig`` object.
        """
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> "LLMConfig":
        """
        Creates an instance from a JSON string.

        :param json_str: JSON-encoded configuration.
        :return:         Initialised ``LLMConfig`` object.
        """
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        """
        Pretty-prints the configuration.

        :return: Human-readable JSON string.
        """
        return json.dumps(self._data, indent=4)


# --------------------------------------------------------------------------- #
#                              Abstract base LLM                              #
# --------------------------------------------------------------------------- #
class BaseLLM(ABC):
    """
    Abstract parent class for any Large-Language-Model implementation.

    Concrete subclasses must implement:
    • ``_init_llm_config`` – extracts runtime parameters from ``global_config``  
    • ``async_infer``      – asynchronous inference  
    • ``infer``            – synchronous single inference  
    • ``batch_infer``      – synchronous batched inference
    """

    global_config: BaseConfig
    llm_name: str
    llm_config: LLMConfig

    # ------------------------------ init ---------------------------------- #
    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        """
        Instantiates a backend-agnostic LLM wrapper.

        :param global_config: Experiment-wide settings. If ``None``, a default
                              ``BaseConfig`` will be created.
        :return:              ``None``.
        """
        if global_config is None:
            logger.debug(
                "Global config not provided — falling back to default BaseConfig."
            )
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        logger.debug(
            "Loading %s with global_config: %s",
            self.__class__.__name__,
            asdict(self.global_config),
        )

        self.llm_name = self.global_config.llm_name
        logger.debug("Set %s.llm_name to: %s", self.__class__.__name__, self.llm_name)

    # ----------------------- subclass responsibilities -------------------- #
    @abstractmethod
    def _init_llm_config(self) -> None:
        """
        Extracts model-specific parameters from ``self.global_config``.

        Implementations **must** assign ``self.llm_config`` and raise an
        exception if any mandatory parameter is missing.

        :return: ``None``.
        """

    # ---------------------------- utilities ------------------------------ #
    def batch_upsert_llm_config(self, updates: Dict[str, Any]) -> None:
        """
        Inserts or overrides multiple fields in ``self.llm_config``.

        :param updates: Mapping of key–value pairs to merge.
        :return:        ``None``.
        """
        self.llm_config.batch_upsert(updates=updates)
        logger.debug(
            "Updated %s.llm_config with %s — resulting config: %s",
            self.__class__.__name__,
            updates,
            self.llm_config,
        )

    # --------------------------- inference API --------------------------- #
    async def async_infer(
        self, messages: List[TextChatMessage], **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        """
        Performs asynchronous inference.

        :param messages: Ordered chat history.
        :param kwargs:   Backend-specific options (e.g. temperature).
        :return:         Pair of (generated messages, metadata).
        """
        pass

    def infer(
        self, messages: List[TextChatMessage], **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        """
        Performs synchronous single-request inference.

        :param messages: Ordered chat history.
        :param kwargs:   Backend-specific options (e.g. temperature).
        :return:         Pair of (generated messages, metadata).
        """
        pass

    def batch_infer(
        self, batch_chat: List[List[TextChatMessage]], **kwargs
    ) -> Tuple[List[List[TextChatMessage]], List[dict]]:
        """
        Performs synchronous batched inference.

        :param batch_chat: List of chat histories, one per element in the batch.
        :param kwargs:     Backend-specific options (e.g. temperature).
        :return:           Tuple of (batch outputs, batch metadata).
        """
        pass