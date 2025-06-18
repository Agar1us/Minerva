import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """
    A centralized dataclass holding all configurable parameters for the application.

    This class serves as a single source of truth for settings, making them
    type-safe and easily accessible throughout the codebase.
    """

    # --- LLM Parameters ---
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "The identifier for the LLM to be used (e.g., 'gpt-4o-mini')."}
    )
    llm_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "The base URL for a self-hosted or alternative OpenAI-compatible API. If None, defaults to the official OpenAI service."}
    )
    llm_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The LLM's API Key for  OpenAI-compatible API."}
    )
    embedding_name: str = field(
        default="text-embedding-3-small",
        metadata={"help": "The identifier for the embedding model to be used."}
    )
    embedding_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "The base URL for a self-hosted embedding model. If None, defaults to the official OpenAI service."}
    )
    embedding_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The Embedding's API Key for  OpenAI-compatible API."}
    )
    azure_endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The Azure Endpoint URI for the LLM. If provided, the system will use the Azure OpenAI service."}
    )
    azure_embedding_endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The Azure Endpoint URI for the embedding model. If provided, uses the Azure OpenAI service."}
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum number of tokens to generate in a single LLM response."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "The number of completion choices to generate for each input message."}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "The random seed for reproducibility in stochastic operations like sampling."}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "The sampling temperature for the LLM. 0.0 means deterministic output."}
    )
    response_format: Optional[Dict] = field(
        default_factory=lambda: {"type": "json_object"},
        metadata={"help": "A dictionary specifying the required output format, e.g., {'type': 'json_object'} to enforce JSON output."}
    )

    # --- Async & Retry Parameters ---
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "The maximum number of times to retry a failing API call."}
    )

    # --- Storage and Caching Parameters ---
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If True, disregards any existing OpenIE results and regenerates them."}
    )
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If True, deletes existing storage and graph data to rebuild the index from scratch."}
    )
    rerank_dspy_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path to a trained DSPy reranker model."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "A multiplicative factor to adjust the weight of passage nodes in Personalized PageRank (PPR)."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If True, saves the results of the OpenIE process to disk."}
    )

    # --- Preprocessing Parameters ---
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "The class name of the text preprocessor to use."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "The model name of the encoder used for token-aware chunking."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "The number of tokens to overlap between adjacent document chunks."}
    )
    preprocess_chunk_max_token_size: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of tokens per chunk. If None, each document is treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(
        default='by_token',
        metadata={"help": "The method for chunking text: 'by_token' or 'by_word'."}
    )

    # --- Information Extraction Parameters ---
    information_extraction_model_name: Literal["openie_openai_gpt",] = field(
        default="openie_openai_gpt",
        metadata={"help": "The class name of the information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "The operational mode for OpenIE: 'online' for real-time API calls, 'offline' for pre-computed batch processing."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "If True, skips the graph construction phase. Useful for pre-generating embeddings with offline models before building the graph."}
    )

    # --- Embedding Parameters ---
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "The identifier of the embedding model to use (e.g., from Hugging Face)."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "The batch size for encoding documents with the embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "If True, normalizes the output embeddings to unit length."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length supported by the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "The data type (precision) for the local embedding model. 'auto' lets the library decide."}
    )

    # --- Graph Construction Parameters ---
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "The number of nearest neighbors (k) to consider when building synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "The batch size for query embeddings during nearest neighbor search for synonymy."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "The batch size for key embeddings during nearest neighbor search for synonymy."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "The similarity threshold above which a synonymy edge is created."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "If True, constructs a directed graph; otherwise, an undirected graph."}
    )

    # --- Retrieval Parameters ---
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of neighboring nodes to explore from a current node during a retrieval step."}
    )
    retrieval_top_k: int = field(
        default=5,
        metadata={"help": "The number of documents/nodes to retrieve at each step of the retrieval process."}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "The damping factor (alpha) for the Personalized PageRank (PPR) algorithm."}
    )

    # --- Question Answering (QA) Parameters ---
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "The maximum number of retrieval-reasoning steps to perform for a single question."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "The number of top-k retrieved documents to feed to the QA model for final answer generation."}
    )

    # --- Directory and Dataset Parameters ---
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The root directory for saving all outputs (models, caches, results). If specified, it overrides all other defaults. If None, it defaults to 'outputs/' for general runs or 'outputs/<dataset_name>/' when a specific dataset is used."}
    )
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default='musique',
        metadata={"help": "The specific dataset to run experiments on. If None, the system operates in a general-purpose mode."}
    )
    graph_type: Literal[
        'dpr_only',
        'entity',
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation',
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "The type of graph structure to build and use for experiments."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "The number of documents from the corpus to use. If None, the entire corpus is used."}
    )

    def __post_init__(self):
        """
        Sets the default save_dir path after initialization if it's not provided.
        """
        if self.save_dir is None:
            if self.dataset is None:
                self.save_dir = 'outputs'
            else:
                self.save_dir = os.path.join('outputs', self.dataset)
        logger.debug(f"Initializing the top-level save directory to: {self.save_dir}")