import logging
import re
from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from .custom_typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                              Data-classes                                   #
# --------------------------------------------------------------------------- #
@dataclass
class NerRawOutput:
    """
    Raw LLM output of a *Named-Entity Recognition* step.

    :param chunk_id:        Unique identifier of the processed chunk.
    :param response:        Original LLM response (raw text / JSON).
    :param unique_entities: Post-processed list of unique entity strings.
    :param metadata:        Extra stats (token counts, cache flag, etc.).
    """

    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    """
    Raw LLM output of a *triple-extraction* step.

    :param chunk_id: Identifier of the processed chunk.
    :param response: Original LLM response (raw text / JSON).
    :param triples:  List of (subject, predicate, object) triples.
    :param metadata: Extra stats (token counts, cache flag, etc.).
    """

    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class LinkingOutput:
    """
    Score vector returned by the retrieval / linking layer.

    :param score: 1-D NumPy array of scores.
    :param type:  Either ``"node"`` (graph node) or ``"dpr"`` (passage DPR).
    """

    score: np.ndarray
    type: Literal["node", "dpr"]


@dataclass
class QuerySolution:
    """
    Bundle that stores the end-to-end QA solution for a single query.

    :param question:    Original question string.
    :param docs:        Ranked list of retrieved document ids.
    :param doc_scores:  NumPy array of retrieval / rerank scores.
    :param full_answer: LLM long-form answer (optional).
    :param short_answer:Short, final answer span (optional).
    :param gold_answers:Ground-truth answers (for eval).
    :param gold_docs:   Gold passage ids (for eval).
    """

    question: str
    docs: List[str]
    doc_scores: Optional[np.ndarray] = None
    full_answer: Optional[str] = None
    short_answer: Optional[str] = None
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the solution to a JSON-serialisable dictionary.

        :return: Dict representation with scores rounded to 4 decimals.
        """
        return {
            "question": self.question,
            "full_answer": self.full_answer,
            "short_answer": self.short_answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": (
                [round(v, 4) for v in self.doc_scores.tolist()[:5]]
                if self.doc_scores is not None
                else None
            ),
            "gold_docs": self.gold_docs,
        }


# --------------------------------------------------------------------------- #
#                              Helper functions                               #
# --------------------------------------------------------------------------- #
def text_processing(text: Any) -> Any:
    """
    Lower-cases and removes non-alphanumeric chars.

    Recursively handles lists.

    :param text: String *or* nested list of strings.
    :return:     Cleaned string or list.
    """
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def reformat_openie_results(
    corpus_openie_results,
) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
    """
    Converts legacy corpus-level OpenIE results to the newer dataclass format.

    :param corpus_openie_results: Iterable where each element is a dict with
                                  keys ``idx``, ``extracted_entities``,
                                  ``extracted_triples``.
    :return: Pair of mappings ``chunk_id -> NerRawOutput`` and
                               ``chunk_id -> TripleRawOutput``.
    """
    ner_dict = {
        item["idx"]: NerRawOutput(
            chunk_id=item["idx"],
            response=None,
            metadata={},
            unique_entities=list(np.unique(item["extracted_entities"])),
        )
        for item in corpus_openie_results
    }
    triple_dict = {
        item["idx"]: TripleRawOutput(
            chunk_id=item["idx"],
            response=None,
            metadata={},
            triples=filter_invalid_triples(item["extracted_triples"]),
        )
        for item in corpus_openie_results
    }
    return ner_dict, triple_dict


def extract_entity_nodes(
    chunk_triples: List[List[Triple]],
) -> Tuple[List[str], List[List[str]]]:
    """
    Aggregates unique entity nodes from triple lists.

    :param chunk_triples: List of per-chunk triple lists.
    :return:              Tuple *(all_unique_entities, per_chunk_entities)*.
    """
    per_chunk: List[List[str]] = []

    for triples in chunk_triples:
        ents = set()
        for t in triples:
            if len(t) == 3:
                ents.update([t[0], t[2]])
            else:
                logger.warning("Invalid triple encountered during graph build: %s", t)
        per_chunk.append(list(ents))

    all_nodes = list(np.unique([e for sub in per_chunk for e in sub]))
    return all_nodes, per_chunk


def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    """
    Deduplicates and flattens triples to a corpus-level fact list.

    :param chunk_triples: List of per-chunk triple lists.
    :return:              List of unique (subject, predicate, object) tuples.
    """
    facts = {tuple(t) for triples in chunk_triples for t in triples}
    return list(set(facts))


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalises an array to the [0, 1] range (min-max scaling).

    :param x: Input NumPy array.
    :return:  Scaled array; returns an all-ones array if *x* is constant.
    """
    min_val, max_val = float(np.min(x)), float(np.max(x))
    if max_val - min_val == 0:
        return np.ones_like(x)
    return (x - min_val) / (max_val - min_val)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Computes ``md5(content)`` and prepends *prefix*.

    :param content: String to hash.
    :param prefix:  Optional prefix (e.g. ``"chunk_"``).
    :return:        Concatenated prefix + hexdigest.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: Dict[Any, List[Any]]) -> bool:
    """
    Checks whether all dict values are sequences of the same length.

    :param data: Dictionary whose values are iterable.
    :return:     ``True`` if equal-length or empty dict, else ``False``.
    """
    iterator = iter(data.values())
    try:
        first_len = len(next(iterator))
    except StopIteration:
        return True
    return all(len(seq) == first_len for seq in iterator)


def string_to_bool(v: Any) -> bool:
    """
    CLI-friendly bool parser.

    Accepts typical textual truthy / falsy variants.

    :param v: Input value (bool or str).
    :return:  Parsed boolean.
    :raises ArgumentTypeError: If the string cannot be interpreted.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "y", "1"}:
        return True
    if v.lower() in {"no", "false", "f", "n", "0"}:
        return False
    raise ArgumentTypeError(
        "Truthy value expected; got %s. "
        "Valid options: yes/no, true/false, t/f, y/n, 1/0 (case-insensitive)." % v
    )