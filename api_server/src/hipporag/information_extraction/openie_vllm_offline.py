import json
from typing import Callable, Dict, List, Tuple

from ..prompts import PromptTemplateManager
from ..information_extraction import OpenIE
from ..llm.vllm_offline import VLLMOffline
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.llm_utils import filter_invalid_triples

from .openie_openai import ChunkInfo

logger = get_logger(__name__)


def _extract_json_field(text: str, field: str) -> List:
    """
    Extracts a top-level array *field* from a JSON string.

    :param text:  Raw JSON string (possibly with surrounding whitespace).
    :param field: Field to extract (e.g. ``"named_entities"`` or ``"triples"``).
    :return:      Parsed list, or an empty one on error.
    """
    try:
        return json.loads(text)[field]
    except Exception as exc:
        logger.warning("Could not parse '%s' from response: %s | response: %s", field, exc, text)
        return []


class VLLMOfflineOpenIE(OpenIE):
    """
    OpenIE pipeline executed via *offline* vLLM batching.
    """
    def __init__(self, global_config) -> None:
        """
        :param global_config: Dict-like object passed to ``VLLMOffline`` backend.
        """
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = VLLMOffline(global_config)

    def _batch_step(
        self,
        *,
        prompt_name: str,
        passages: List[str],
        field: str,
        json_template: str,
        postprocess: Callable[[List], List],
        template_kwargs: List[Dict] | None = None,
        max_tokens: int = 512,
    ) -> Tuple[List[str], List[List]]:
        """
        Runs a single vLLM offline batch (NER or triple extraction).

        :param prompt_name:   Name of the prompt template to render.
        :param passages:      List of passages to process.
        :param field:         JSON field expected in LLM output.
        :param json_template: Name passed to ``batch_infer`` for parsing.
        :param postprocess:   Callable applied to each list value (e.g. dedup).
        :param template_kwargs: Per-passage extra kwargs forwarded to renderer.
        :param max_tokens:    Decoder budget for the backend model.
        :return:              Tuple ``(raw_responses, processed_items)``
                              where *processed_items* is a list of lists.
        """
        template_kwargs = template_kwargs or [{} for _ in passages]

        input_messages = [
            self.prompt_template_manager.render(name=prompt_name, passage=passage, **kw)
            for passage, kw in zip(passages, template_kwargs, strict=True)
        ]

        raw_responses, _meta = self.llm_model.batch_infer(
            input_messages, json_template=json_template, max_tokens=max_tokens
        )

        items: List[List] = []
        for raw in raw_responses:
            extracted = _extract_json_field(raw, field)
            items.append(postprocess(extracted))

        return raw_responses, items

    def batch_openie(
        self,
        chunks: Dict[str, ChunkInfo],
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Runs NER and triple extraction on a batch of *chunks*.

        :param chunks: Mapping ``chunk_id -> ChunkInfo`` to process.
        :return:       Two dictionaries:
                       ``chunk_id -> NerRawOutput`` and
                       ``chunk_id -> TripleRawOutput``.
        """
        chunk_ids: List[str] = list(chunks.keys())
        passages: List[str] = [chunks[cid]["content"] for cid in chunk_ids]

        # -------- NER step ------------------------------------------------
        ner_raw, named_entities_batch = self._batch_step(
            prompt_name="ner",
            passages=passages,
            field="named_entities",
            json_template="ner",
            postprocess=lambda xs: list(dict.fromkeys(xs)),
            max_tokens=512,
        )

        ner_results: Dict[str, NerRawOutput] = {
            cid: NerRawOutput(
                chunk_id=cid,
                response=resp,
                unique_entities=ents,
                metadata={},
            )
            for cid, resp, ents in zip(chunk_ids, ner_raw, named_entities_batch, strict=True)
        }

        # -------- Triple extraction step ----------------------------------
        kw_list = [
            {"named_entity_json": json.dumps({"named_entities": ents})}
            for ents in named_entities_batch
        ]

        triple_raw, triples_batch = self._batch_step(
            prompt_name="triple_extraction",
            passages=passages,
            field="triples",
            json_template="triples",
            postprocess=lambda xs: filter_invalid_triples(triples=xs),
            template_kwargs=kw_list,
            max_tokens=2048,
        )

        triple_results: Dict[str, TripleRawOutput] = {
            cid: TripleRawOutput(
                chunk_id=cid,
                response=resp,
                triples=triples,
                metadata={},
            )
            for cid, resp, triples in zip(chunk_ids, triple_raw, triples_batch, strict=True)
        }

        # -------- Logging helpers -----------------------------------------
        for cid, ents in zip(chunk_ids, named_entities_batch, strict=True):
            if not ents:
                logger.warning("No entities extracted for chunk_id: %s", cid)
        for cid, triples in zip(chunk_ids, triples_batch, strict=True):
            if not triples:
                logger.warning("No triples extracted for chunk_id: %s", cid)

        return ner_results, triple_results