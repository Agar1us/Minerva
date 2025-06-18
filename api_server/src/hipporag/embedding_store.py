import os
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
import fireducks.pandas as pd

from .utils.misc_utils import compute_mdhash_id
from .utils.logging_utils import get_logger
from .embedding_model import BaseEmbeddingModel

logger = get_logger(__name__)


class EmbeddingStore:
    """
    A class to manage storing, retrieving, and embedding text data persistently.

    This store uses a Parquet file for on-disk storage and a pandas DataFrame
    for in-memory operations. It automatically handles hashing, embedding of new texts,
    and caching to avoid re-computing embeddings for existing data.
    """

    def __init__(
        self, embedding_model: BaseEmbeddingModel, db_dir: str, namespace: str
    ) -> None:
        """
        Initializes the EmbeddingStore.

        :param embedding_model: The embedding model for encoding texts
        :param db_dir: The directory path where the data file will be stored.
        :param namespace: A unique string to isolate data within the store, used as a prefix for hashes.
        """
        self.embedding_model = embedding_model
        self.namespace = namespace
        self.db_dir = db_dir

        os.makedirs(self.db_dir, exist_ok=True)
        self.filename = os.path.join(self.db_dir, f"vdb_{self.namespace}.parquet")

        self._load_data()

    def _load_data(self) -> None:
        """
        Loads data from the Parquet file into a pandas DataFrame.
        If the file does not exist, an empty DataFrame is created.
        """
        if os.path.exists(self.filename):
            self.df = pd.read_parquet(self.filename).set_index("hash_id")
            logger.info(f"Loaded {len(self.df)} records from {self.filename}")
        else:
            self.df = pd.DataFrame(
                columns=["hash_id", "content", "embedding"]
            ).set_index("hash_id")

    def _save_data(self) -> None:
        """Saves the current state of the DataFrame to the Parquet file."""
        self.df.reset_index().to_parquet(self.filename, index=False)
        self._clear_cache()
        logger.info(f"Saved {len(self.df)} records to {self.filename}")

    def _clear_cache(self) -> None:
        """Clears all cached properties to force re-computation on next access."""
        for key in ("hash_id_to_content", "content_to_hash_id"):
            if key in self.__dict__:
                del self.__dict__[key]

    @staticmethod
    def _generate_hash_id(text: str, namespace: str) -> str:
        """
        Generates a unique, prefixed MD5 hash for a given text.

        :param text: The input string.
        :param namespace: The namespace prefix.
        :return: A unique hash ID string.
        """
        return compute_mdhash_id(text, prefix=f"{namespace}-")

    @cached_property
    def hash_id_to_content(self) -> Dict[str, str]:
        """A cached dictionary mapping hash IDs to their text content."""
        return self.df["content"].to_dict()

    @cached_property
    def content_to_hash_id(self) -> Dict[str, str]:
        """A cached dictionary mapping text content to their hash IDs."""
        return {v: k for k, v in self.hash_id_to_content.items()}

    def __len__(self) -> int:
        """Returns the number of records in the store."""
        return len(self.df)

    def filter_new_texts(self, texts: List[str]) -> List[str]:
        """
        Identifies which texts from the input list are not already in the store.

        :param texts: A list of texts to check.
        :return: A list containing only the texts that are new to the store.
        """
        existing_contents = set(self.content_to_hash_id.keys())
        return [text for text in texts if text not in existing_contents]

    def add(self, texts: List[str]) -> None:
        """
        Adds a list of new texts to the store.

        This method filters out texts that already exist, embeds the new ones,
        and saves them to the store.

        :param texts: A list of texts to add.
        """
        new_texts = self.filter_new_texts(texts)
        if not new_texts:
            logger.info("No new texts to add. All provided texts already exist in the store.")
            return

        logger.info(f"Found {len(new_texts)} new texts to embed and add to the store.")
        new_embeddings = self.embedding_model.batch_encode(new_texts)
        new_hash_ids = [self._generate_hash_id(text, self.namespace) for text in new_texts]

        new_data = pd.DataFrame(
            {"content": new_texts, "embedding": list(new_embeddings)},
            index=pd.Index(new_hash_ids, name="hash_id"),
        )
        
        self.df = pd.concat([self.df, new_data])
        self._save_data()

    def delete(self, hash_ids: List[str]) -> None:
        """
        Deletes records from the store by their hash IDs.

        :param hash_ids: A list of hash IDs to delete.
        """
        original_count = len(self)
        self.df.drop(index=hash_ids, inplace=True, errors="ignore")
        deleted_count = original_count - len(self)
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} records. Saving changes.")
            self._save_data()
        else:
            logger.info("No records were deleted (provided IDs not found).")

    def get_hash_id(self, text: str) -> Optional[str]:
        """
        Retrieves the hash ID for a given text.

        :param text: The text to look up.
        :return: The corresponding hash ID, or None if the text is not in the store.
        """
        return self.content_to_hash_id.get(text)
    
    def get_content(self, hash_id: str) -> Optional[str]:
        """
        Retrieves the text content for a given hash ID.

        :param hash_id: The hash ID to look up.
        :return: The corresponding text content, or None if the ID is not found.
        """
        return self.hash_id_to_content.get(hash_id)

    def get_all_ids(self) -> List[str]:
        """Returns a list of all hash IDs in the store."""
        return self.df.index.tolist()

    def get_rows_content(self, ids: List[str]) -> List[str]:
        """
        Returns a row with current hash_id
        
        :param ids: The hash_ids of requested rows
        :return: A pandas.Series with requested hash_ids
        """
        contents = self.df.loc[ids]['content'].to_list()
        if len(contents) == 1:
            contents = contents[0]
        return contents

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, str]]:
        """
        Returns a dictionary mapping each hash ID to a row-like dictionary.

        :return: A dictionary in the format: `{hash_id: {'hash_id': hash_id, 'content': content}}`.
        """
        return {
            hash_id: {"hash_id": hash_id, "content": content}
            for hash_id, content in self.hash_id_to_content.items()
        }

    def get_all_embeddings(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Returns all embeddings as a single NumPy array.

        :param dtype: The desired data type of the output array.
        :return: A NumPy array of all embeddings.
        """
        if self.df.empty:
            return np.array([], dtype=dtype)
        return np.vstack(self.df["embedding"].values).astype(dtype)

    def get_embeddings(self, hash_ids: List[str], dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
        """
        Retrieves embeddings for a specific list of hash IDs.

        :param hash_ids: A list of hash IDs to retrieve embeddings for.
        :param dtype: The desired data type of the output array.
        :return: A NumPy array of the requested embeddings in the same order as the input IDs.
                 Returns None if the store or the input list is empty.
        """
        if not hash_ids or self.df.empty:
            return None
            
        embeddings_series = self.df.loc[hash_ids, "embedding"]
        return np.vstack(embeddings_series.values).astype(dtype)
