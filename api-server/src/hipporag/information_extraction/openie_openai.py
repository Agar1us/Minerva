import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict

from tqdm.asyncio import tqdm as async_tqdm

from ..llm.openai_gpt import CacheOpenAI
from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import NerRawOutput, TripleRawOutput

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# ============================ Type definitions ============================= #
# --------------------------------------------------------------------------- #


class ChunkInfo(TypedDict):
    """
    Holds metadata for a document chunk.

    :param num_tokens:   Number of tokens inside the chunk.
    :param content:      Raw text content of the chunk.
    :param chunk_order:  Position markers (e.g. page, paragraph, sentence).
    :param full_doc_ids: IDs of documents this chunk belongs to.
    """
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    """
    Helper wrapper for batching LLM calls.

    :param chunk_id:      Hash / id of the chunk.
    :param input_message: List of messages passed to the LLM.
    """

    chunk_id: str
    input_message: List[Dict]


# --------------------------------------------------------------------------- #
# ============================= Helper functions ============================ #
# --------------------------------------------------------------------------- #

_JSON_OBJECT_RGX = re.compile(
    r'\{[^{}]*"(?P<field>[^"]+)"\s*:\s*\[[^\]]*]\s*[^{}]*}', re.DOTALL
)


def _extract_field_from_response(response: str, field: str, warn: bool = True) -> List:
    """
    Extracts an array field (e.g. ``named_entities`` or ``triples``) from a raw
    LLM response that may contain extra text.

    :param response: Raw text returned by the LLM.
    :param field:    Target JSON key to extract.
    :param warn:     Whether to log a warning on failure.
    :return:         A list with the extracted items or an empty list.
    """
    match = _JSON_OBJECT_RGX.search(response)
    if not match or match.group("field") != field:
        if warn:
            logger.warning("Could not find field '%s' in LLM response: %s", field, response)
        return []

    try:
        return json.loads(match.group())[field]
    except (json.JSONDecodeError, KeyError) as exc:
        if warn:
            logger.warning("Failed to parse JSON for field '%s': %s | response: %s", field, exc, response)
        return []


# --------------------------------------------------------------------------- #
# ================================= OpenIE ================================== #
# --------------------------------------------------------------------------- #


class OpenIE:
    """
    High-level OpenIE pipeline — first NER, then triple extraction.

    :param llm_model: An LLM wrapper exposing ``async_infer`` with OpenAI-style semantics.
    """

    def __init__(self, llm_model: CacheOpenAI) -> None:
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    # --------------------------------------------------------------------- #
    # =============================== NER ================================= #
    # --------------------------------------------------------------------- #

    async def ner(self, chunk_id: str, passage: str) -> NerRawOutput:
        """
        Performs Named-Entity Recognition on a passage.

        :param chunk_id: Identifier of the chunk being processed.
        :param passage:  Text passage to analyse.
        :return:         ``NerRawOutput`` that bundles raw response, entities and metadata.
        """
        messages = self.prompt_template_manager.render(name="ner", passage=passage)
        raw_response: str = ""
        meta: Dict = {}

        try:
            raw_response, meta = await self.llm_model.async_infer(
                messages=messages, response_format={"type": "json_object"}
            )

            real_response = (
                fix_broken_generated_json(raw_response)
                if meta.get("finish_reason") == "length"
                else raw_response
            )
            entities = _extract_field_from_response(real_response, "named_entities")
            unique_entities = list(dict.fromkeys(entities))
        except Exception as exc:
            logger.warning("NER failed for chunk %s: %s", chunk_id, exc)
            meta["error"] = str(exc)
            unique_entities = []

        return NerRawOutput(
            chunk_id=chunk_id,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=meta,
        )

    # --------------------------------------------------------------------- #
    # ======================== Triple extraction ========================== #
    # --------------------------------------------------------------------- #

    async def triple_extraction(
        self, chunk_id: str, passage: str, named_entities: List[str]
    ) -> TripleRawOutput:
        """
        Extracts (subject, predicate, object) triples from a passage.

        :param chunk_id:       Identifier of the chunk being processed.
        :param passage:        Text passage to analyse.
        :param named_entities: Entities already detected by NER.
        :return:               ``TripleRawOutput`` with extracted triples and metadata.
        """
        messages = self.prompt_template_manager.render(
            name="triple_extraction",
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities}),
        )

        raw_response: str = ""
        meta: Dict = {}

        try:
            raw_response, meta = await self.llm_model.async_infer(
                messages=messages, response_format={"type": "json_object"}
            )

            real_response = (
                fix_broken_generated_json(raw_response)
                if meta.get("finish_reason") == "length"
                else raw_response
            )
            triples = _extract_field_from_response(real_response, "triples")
            triples = filter_invalid_triples(triples=triples)
        except Exception as exc:
            logger.warning("Triple extraction failed for chunk %s: %s", chunk_id, exc)
            meta["error"] = str(exc)
            triples = []

        return TripleRawOutput(
            chunk_id=chunk_id,
            response=raw_response,
            metadata=meta,
            triples=triples,
        )

    # --------------------------------------------------------------------- #
    # ======================= Single-chunk pipeline ======================= #
    # --------------------------------------------------------------------- #

    async def _process_chunk(self, chunk_id: str, passage: str) -> Tuple[NerRawOutput, TripleRawOutput]:
        """
        Runs NER and then triple extraction for a single chunk.

        :param chunk_id: Identifier of the chunk.
        :param passage:  Text passage to process.
        :return:         Tuple of ``(NerRawOutput, TripleRawOutput)``.
        """
        ner_out = await self.ner(chunk_id=chunk_id, passage=passage)
        triple_out = await self.triple_extraction(
            chunk_id=chunk_id, passage=passage, named_entities=ner_out.unique_entities
        )
        return ner_out, triple_out

    # --------------------------------------------------------------------- #
    # ============================= Batching ============================== #
    # --------------------------------------------------------------------- #

    async def batch_openie(
        self, chunks: Dict[str, ChunkInfo]
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Asynchronously runs the OpenIE pipeline on multiple chunks.

        :param chunks: Mapping ``chunk_id -> ChunkInfo`` to process.
        :return:       Tuple of two dicts —
                       ``chunk_id -> NerRawOutput`` and ``chunk_id -> TripleRawOutput``.
        """
        chunk_passages = {cid: info["content"] for cid, info in chunks.items()}
        tasks = [self._process_chunk(cid, txt) for cid, txt in chunk_passages.items()]

        ner_results: Dict[str, NerRawOutput] = {}
        triple_results: Dict[str, TripleRawOutput] = {}

        total_prompt_tokens = 0
        total_completion_tokens = 0
        cache_hits = 0

        pbar = async_tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="OpenIE: NER + triples"
        )
        async for fut in pbar:
            ner_out, triple_out = await fut

            ner_results[ner_out.chunk_id] = ner_out
            triple_results[triple_out.chunk_id] = triple_out

            for meta in (ner_out.metadata, triple_out.metadata):
                total_prompt_tokens += meta.get("prompt_tokens", 0)
                total_completion_tokens += meta.get("completion_tokens", 0)
                cache_hits += int(bool(meta.get("cache_hit")))

            pbar.set_postfix(
                {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "cache_hits": cache_hits,
                }
            )

        return ner_results, triple_results