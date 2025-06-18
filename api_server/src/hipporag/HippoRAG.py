import json
import os
import logging
from dataclasses import asdict
from typing import List, Set, Dict, Tuple
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial
import igraph as ig
import numpy as np
from collections import defaultdict
import pickle
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import OpenAIEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import LLMFactReranker
from .utils.misc_utils import *
from .utils.embed_utils import retrieve_knn
from .utils.config_utils import BaseConfig
from .utils.llm_utils import fix_broken_generated_json
from .utils.text_extractor import collect_texts

logger = logging.getLogger(__name__)

class HippoRAG:
    """
    The main orchestrator class for the HippoRAG framework.

    This class integrates all components, including language models, embedding models,
    information extraction, and graph construction to build and query a knowledge graph.
    """

    def __init__(
        self,
        global_config: BaseConfig | None = None,
        save_dir: str | None = None,
        llm_name: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_name: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
    ):
        """
        Initializes the HippoRAG system.

        :param global_config: A BaseConfig object. If None, a default config is created.
        :param save_dir: Overrides the save directory in the config.
        :param llm_model_name: Overrides the LLM model name.
        :param llm_base_url: Overrides the LLM base URL.
        :param llm_api_key: Overrides the LLM API key.
        :param embedding_model_name: Overrides the embedding model name.
        :param embedding_base_url: Overrides the embedding base URL.
        :param embedding_api_key: Overrides the embedding API key.
        """
        self._setup_config(global_config, locals())
        self._setup_paths()
        self._initialize_components()

    def _setup_config(self, config: BaseConfig | None, overrides: Dict[str, Any]) -> None:
        """
        Initializes the global configuration, applying any overrides from the constructor.
        """
        self.global_config = config or BaseConfig()
        
        for key, value in overrides.items():
            if key not in ("self", "config") and value is not None:
                setattr(self.global_config, key, value)
        
        config_str = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG initialized with config:\n  {config_str}\n")

    def _setup_paths(self) -> None:
        """Sets up all necessary directories and file paths based on the configuration."""
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_name.replace("/", "_")
        
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.openie_results_path = os.path.join(
            self.working_dir, f'openie_results_{llm_label}.json'
        )
        self.graph_path = os.path.join(self.working_dir, "graph.pickle")
        self.node_chunk_map_path = os.path.join(self.working_dir, "node_to_chunk_map.pickle")
        logger.info(f"Working directory set to: {self.working_dir}")

    def _initialize_components(self) -> None:
        """Initializes all the major components of the HippoRAG system."""
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        
        if self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
            self.embedding_model = None
        else:
            self.openie = OpenIE(llm_model=self.llm_model)
            self.embedding_model = OpenAIEmbeddingModel(global_config=self.global_config)

        self.chunk_embedding_store = EmbeddingStore(self.embedding_model, self.working_dir, "chunk")
        self.entity_embedding_store = EmbeddingStore(self.embedding_model, self.working_dir, "entity")
        self.fact_embedding_store = EmbeddingStore(self.embedding_model, self.working_dir, "fact")

        self.rerank_filter = LLMFactReranker(self)
        self.graph = self._load_or_initialize_graph()
        self.ent_node_to_chunk_ids = self._load_node_chunk_map()
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.ready_to_retrieve = False
        self.df_lock = threading.RLock() # Требуется для работы с fireducks

    def _load_or_initialize_graph(self) -> ig.Graph:
        """Loads the graph from a file or creates a new one if not found."""
        if not self.global_config.force_index_from_scratch and os.path.exists(self.graph_path):
            logger.info(f"Loading graph from {self.graph_path}")
            graph = ig.Graph.Read_Pickle(self.graph_path)
            logger.info(f"Loaded graph with {graph.vcount()} nodes and {graph.ecount()} edges.")
            return graph
        
        logger.info("Initializing a new graph.")
        return ig.Graph(directed=self.global_config.is_directed_graph)
    
    def _load_node_chunk_map(self) -> Dict[str, List[str]]:
        """Loads the entity-to-chunk mapping from a file."""
        if os.path.exists(self.node_chunk_map_path):
            with open(self.node_chunk_map_path, "rb") as f:
                return pickle.load(f)
        return defaultdict(list)

    def thread_safe_get_rows_content(self, store: str, keys):
        """Thread-safe обертка для доступа к DataFrame"""
        with self.df_lock:
            if store == 'chunk':
                return self.chunk_embedding_store.get_rows_content(keys)
            else:
                return self.fact_embedding_store.get_rows_content(keys)

    async def index(self, docs: List[str]):
        """
        Indexes documents by generating a knowledge graph and encoding passages, entities, and facts.

        The process involves:
        1. Performing OpenIE to extract entities and triples from documents.
        2. Storing and encoding the document chunks (passages).
        3. Storing and encoding the extracted entities and facts.
        4. Constructing a heterogeneous graph with different types of nodes and edges.
        5. Augmenting the graph with synonymy edges.
        6. Saving the final graph and associated mappings.

        :param docs: A list of document file paths or raw text content to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")
        docs = collect_texts(docs)
        self.chunk_embedding_store.add(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = await self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)
        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.add(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.add([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes specified documents and their associated data from the system.

        This method removes document chunks, and any facts and entities that are
        exclusively derived from those chunks. It updates the embedding stores and
        rebuilds the graph to reflect the deletions.

        :param docs_to_delete: A list of document contents to be deleted.
        """

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    async def rag_qa(self,
               queries: List[str|QuerySolution],
               ) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs Retrieval-Augmented Generation (RAG) for a list of queries.

        This is a high-level convenience method that chains the `retrieve` and `qa` steps.
        It first retrieves relevant documents for the queries and then generates answers
        based on the retrieved context.

        :param queries: A list of query strings or `QuerySolution` objects.
        :return: A tuple containing:
                 - A list of `QuerySolution` objects with retrieved docs and generated answers.
                 - A list of raw response messages from the language model.
                 - A list of metadata dictionaries for each QA call.
        """

        if not isinstance(queries[0], QuerySolution):
            queries = await self.retrieve(queries=queries)
        queries_solutions, all_response_message, all_metadata = await self.qa(queries)
        return queries_solutions, all_response_message, all_metadata

    async def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Retrieves relevant documents for a list of queries using a multi-stage process with parallel execution.

        This method performs document retrieval through a sophisticated pipeline that includes
        query embedding generation, fact scoring, reranking, and graph-based search. The process
        runs concurrently for multiple queries to optimize performance.

        :param queries: A list of query strings to retrieve documents for.
        :param num_to_retrieve: Number of top documents to retrieve per query. If None, uses global config value.
        :return: A list of `QuerySolution` objects, each containing a query with its retrieved documents and scores.
        """
        retrieve_start_time = time.time()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        loop = asyncio.get_event_loop()
        
        max_workers = min(len(queries), 12)
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fact_scores_tasks = [
                loop.run_in_executor(executor, self.get_fact_scores, query)
                for query in queries
            ]
            all_fact_scores = await asyncio.gather(*fact_scores_tasks)

            rerank_start = time.time()
            rerank_tasks = [
                self.rerank_facts(query, fact_scores)
                for query, fact_scores in zip(queries, all_fact_scores)
            ]
            rerank_results = await asyncio.gather(*rerank_tasks)
            rerank_end = time.time()
            rerank_time = rerank_end - rerank_start

            search_tasks = []
            for query, fact_scores, (top_k_fact_indices, top_k_facts, rerank_log) in zip(
                queries, all_fact_scores, rerank_results
            ):
                if len(top_k_facts) == 0:
                    logger.info('No facts found after reranking, return DPR results')
                    task = loop.run_in_executor(executor, self.dense_passage_retrieval, query)
                else:
                    search_func = partial(
                        self.graph_search_with_fact_entities,
                        query=query,
                        link_top_k=self.global_config.linking_top_k,
                        query_fact_scores=fact_scores,
                        top_k_facts=top_k_facts,
                        top_k_fact_indices=top_k_fact_indices,
                        passage_node_weight=self.global_config.passage_node_weight
                    )
                    task = loop.run_in_executor(executor, search_func)
                search_tasks.append(task)
            
            search_results = await asyncio.gather(*search_tasks)

        retrieval_results = []
        for query, (sorted_doc_ids, sorted_doc_scores) in (zip(queries, search_results)):
            top_k_docs = [
                self.thread_safe_get_rows_content(store='chunk', keys=[self.passage_node_keys[idx]]) 
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]
            retrieval_results.append(
                QuerySolution(
                    question=query, 
                    docs=top_k_docs, 
                    doc_scores=sorted_doc_scores[:num_to_retrieve]
                )
            )

        retrieve_end_time = time.time()
        all_retrieval_time = retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {rerank_time:.2f}s")

        return retrieval_results

    async def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Generates answers for a list of queries with their retrieved contexts.

        This method concurrently processes each `QuerySolution` object, feeding the question
        and its retrieved documents to the language model to generate an answer.

        :param queries: A list of `QuerySolution` objects, each with a question and retrieved documents.
        :return: A tuple containing:
                 - The updated list of `QuerySolution` objects with answers.
                 - A list of raw LLM response messages.
                 - A list of metadata dictionaries from the LLM calls.
        """
        if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
            prompt_dataset_name = self.global_config.dataset
        else:
            logger.debug(f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
            prompt_dataset_name = 'musique'
        
        tasks = [self.qa_one(query_solution, prompt_dataset_name) for query_solution in queries]
        
        results = await tqdm_asyncio.gather(*tasks, desc="QA Reading")

        if not results:
            return [], [], []

        queries_solutions, all_response_message, all_metadata = zip(*results)
        
        return list(queries_solutions), list(all_response_message), list(all_metadata)

    async def qa_one(self, query_solution: QuerySolution, prompt_dataset_name: str) -> Tuple[QuerySolution, str, Dict]:
        """
        Generates an answer for a single query using its retrieved documents.

        It constructs a prompt from the documents, sends it to the LLM, and parses
        the full and short answers from the JSON response.

        :param query_solution: The `QuerySolution` object containing the question and documents.
        :param prompt_dataset_name: The name of the prompt template to use.
        :return: A tuple containing the updated `QuerySolution`, the raw LLM response, and metadata.
        """
        def extract_answers_from_response(response):
            pattern = r'\{[^{}]*"full_answer"\s*:\s*("[^"]*"|\[[^\]]*\])[^{}]*"short_answer"\s*:\s*("[^"]*"|\[[^\]]*\])[^{}]*\}'
            match = re.search(pattern, response, re.DOTALL)
            if not match:
                logger.warning(f"Could not extract answers from response: {response}")
                return (None, None)
            try:
                data = json.loads(match.group())
                full_answer = data.get("full_answer")
                short_answer = data.get("short_answer")
                return (full_answer, short_answer)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing answers JSON: {e}. Response was: {response}")
                return (None, None)
    
        retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]
        prompt_user = ''
        for passage in retrieved_passages:
            prompt_user += f'Wikipedia Title: {passage}\n\n'
        prompt_user += 'Question: ' + query_solution.question + '\nThought: '
        qa_messages = self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user)
        response_message, metadata = await self.llm_model.async_infer(messages=qa_messages, response_format={'type': 'json_object'})
        try:
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(response_message)
            else:
                real_response = response_message
            full_answer, short_answer = extract_answers_from_response(real_response)
        except Exception as e:
            logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
            full_answer, short_answer = response_message, ''
        query_solution.full_answer = full_answer
        query_solution.short_answer = short_answer
        return query_solution, response_message, metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        :param chunk_ids: A list of unique identifiers for the chunks.
        :param chunk_triples: A list of lists, where each inner list contains the triples
                              extracted from the corresponding chunk.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        :param chunk_ids: A list of identifiers for passage nodes.
        :param chunk_triple_entities: A list of lists, where each inner list contains
                                      the entities found in the corresponding chunk.
        :return: The number of new passage nodes added.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.
        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        :param chunk_keys: A list of all current chunk keys in the system.
        :return: A tuple containing:
                 - A list of all existing OpenIE information.
                 - A set of chunk keys that are new and require processing.
        """

        chunk_keys_to_save = set()
        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])
            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        :param all_openie_info: The list to append new results to.
        :param chunks_to_save: A dictionary of new chunks that were processed.
        :param ner_results_dict: The NER results for the new chunks.
        :param triple_results_dict: The triple extraction results for the new chunks.
        :return: The updated list of all OpenIE information.
        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        :param all_openie_info: A list of dictionaries, each containing OpenIE results for a document.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        """
        Saves the current state of the graph to a pickle file.
        """
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self.graph_path)
        with open(self.node_chunk_map_path, "wb") as f:
            pickle.dump(self.ent_node_to_chunk_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        :return: A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) 
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                self.add_new_nodes()
                self.save_igraph()
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys]
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        :param queries: A list of query strings or `QuerySolution` objects.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        :param query: The input query string.
        :return: A normalized NumPy array of similarity scores.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                    query,
                    instruction=get_query_instruction('query_to_fact'),
                    norm=True
                )

        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        :param query: The input query string.
        :return: A tuple containing:
                 - An array of document indices sorted by relevance.
                 - An array of corresponding normalized similarity scores.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        :param link_top_k: The number of top-ranked phrases to keep.
        :param all_phrase_weights: An array of weights for all phrases.
        :param linking_score_map: A map from phrase content to its linking score.
        :return: A tuple containing:
                 - The filtered phrase weights array (others set to 0).
                 - The filtered linking score map for the top-k phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        :param query: The user query.
        :param link_top_k: The number of top facts/entities to use for personalization.
        :param query_fact_scores: Similarity scores for all facts.
        :param top_k_facts: The list of top-k facts after reranking.
        :param top_k_fact_indices: The original indices of the top-k facts.
        :param passage_node_weight: The weight to assign to the dense retrieval signal.
        :return: A tuple of sorted document indices and their PPR scores.
        """

        linking_score_map = {}
        phrase_scores = {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)

        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.thread_safe_get_rows_content(store='chunk', keys=[passage_node_key])
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        node_weights = phrase_weights + passage_weights

        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    async def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            candidate_facts = self.thread_safe_get_rows_content(store='fact', keys=real_candidate_fact_ids)
            candidate_facts = [eval(fact) for fact in candidate_facts]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = await self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        :param reset_prob: A NumPy array with the reset probabilities for each node.
        :param damping: The damping factor (alpha) for PageRank.
        :return: A tuple of sorted passage node indices and their PPR scores.
        """

        if damping is None: damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
