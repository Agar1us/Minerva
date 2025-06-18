from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .logging_utils import get_logger

logger = get_logger(__name__)


def _get_batches(
    data: torch.Tensor, batch_size: int
) -> Generator[Tuple[torch.Tensor, int], None, None]:
    """
    A generator that yields batches of data from a tensor.

    :param data: The tensor to be batched.
    :param batch_size: The size of each batch.
    :yield: A tuple containing the data batch (torch.Tensor) and the starting index of the batch.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size], i


def retrieve_knn(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs: np.ndarray,
    key_vecs: np.ndarray,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000,
) -> Dict[str, Tuple[List[str], List[float]]]:
    """
    Retrieves the top-k nearest neighbors for each query vector from a set of key vectors.

    This function performs an exhaustive k-NN search using cosine similarity, optimized
    for large datasets by batching both queries and keys to fit within GPU memory.

    :param query_ids: A list of unique identifiers for each query vector.
    :param key_ids: A list of unique identifiers for each key vector.
    :param query_vecs: A NumPy array of query vectors, shape (num_queries, dim).
    :param key_vecs: A NumPy array of key vectors, shape (num_keys, dim).
    :param k: The number of nearest neighbors to retrieve for each query.
    :param query_batch_size: The number of queries to process in a single batch.
    :param key_batch_size: The number of keys to process in a single batch against each query batch.
    :return: A dictionary mapping each query ID to a tuple containing two lists:
             - The list of top-k key IDs.
             - The list of corresponding similarity scores.
    """
    if len(key_vecs) == 0:
        return {qid: ([], []) for qid in query_ids}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Performing KNN search on device: {device}")

    queries = torch.from_numpy(query_vecs).to(torch.float32)
    queries = torch.nn.functional.normalize(queries, dim=1)
    keys = torch.from_numpy(key_vecs).to(torch.float32)
    keys = torch.nn.functional.normalize(keys, dim=1)

    results: Dict[str, Tuple[List[str], List[float]]] = {}

    with torch.no_grad():
        for query_batch, query_start_idx in tqdm(
            _get_batches(queries, query_batch_size),
            total=(len(queries) + query_batch_size - 1) // query_batch_size,
            desc="Finding KNN for Queries",
        ):
            query_batch_gpu = query_batch.to(device)

            batch_top_scores = torch.full((query_batch.size(0), k), -1.0, device=device)
            batch_top_indices = torch.full((query_batch.size(0), k), -1, dtype=torch.long, device=device)

            for key_batch, key_start_idx in _get_batches(keys, key_batch_size):
                key_batch_gpu = key_batch.to(device)

                similarity = torch.mm(query_batch_gpu, key_batch_gpu.T)

                combined_scores = torch.cat([batch_top_scores, similarity], dim=1)
                
                global_key_indices = torch.arange(
                    key_start_idx, key_start_idx + key_batch.size(0), device=device
                ).expand(query_batch.size(0), -1)
                
                combined_indices = torch.cat([batch_top_indices, global_key_indices], dim=1)

                batch_top_scores, top_indices_in_combined = torch.topk(
                    combined_scores, k=min(k, combined_scores.size(1)), dim=1
                )
                
                batch_top_indices = torch.gather(combined_indices, 1, top_indices_in_combined)

            batch_top_scores = batch_top_scores.cpu()
            batch_top_indices = batch_top_indices.cpu()

            for i in range(query_batch.size(0)):
                global_query_idx = query_start_idx + i
                query_id = query_ids[global_query_idx]
                
                valid_indices_mask = batch_top_indices[i] != -1
                
                top_key_indices = batch_top_indices[i][valid_indices_mask].numpy()
                top_key_scores = batch_top_scores[i][valid_indices_mask].numpy()
                
                sorted_order = np.argsort(top_key_scores)[::-1]
                
                final_key_ids = [key_ids[idx] for idx in top_key_indices[sorted_order]]
                final_key_scores = top_key_scores[sorted_order].tolist()

                results[query_id] = (final_key_ids, final_key_scores)

    return results