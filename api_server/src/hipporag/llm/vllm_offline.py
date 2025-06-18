import os
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import PROMPT_JSON_TEMPLATE, TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


class VLLMOffline(BaseLLM):
    """
    A wrapper for running inference with vLLM in an offline, synchronous manner.
    """

    def __init__(self, global_config: BaseConfig) -> None:
        """
        Initializes the vLLM engine and tokenizer.

        :param global_config: The global configuration object for the run.
        :return: None.
        """
        super().__init__(global_config)
        self._init_llm_config()

        # Set environment variable needed for vLLM multiprocessing
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        vllm_params = self._configure_vllm_params()
        self.client = LLM(**vllm_params)
        self.tokenizer = self.client.get_tokenizer()
        logger.info(f"Initialized vLLM with model: {self.llm_config.model_name}")

    def _init_llm_config(self) -> None:
        """
        Loads vLLM-specific configuration from the global config.
        """
        self.llm_config = LLMConfig.from_dict({
            "model_name": self.global_config.get("llm_name", "meta-llama/Llama-3.3-70B-Instruct"),
            "num_gpus": self.global_config.get("num_gpus", torch.cuda.device_count()),
            "max_model_len": self.global_config.get("max_model_len", 4096),
            "seed": self.global_config.get("seed", 0),
            "gpu_memory_utilization": self.global_config.get("gpu_memory_utilization", 0.8),
            "quantization": self.global_config.get("quantization", None),
            "load_format": self.global_config.get("load_format", "auto"),
        })

    def _configure_vllm_params(self) -> Dict:
        """
        Prepares the dictionary of parameters for vLLM engine initialization.

        This method centralizes the logic for determining tensor parallelism,
        quantization methods, and other hardware-specific settings.

        :return: A dictionary of parameters for the `vllm.LLM` constructor.
        """
        model_name = self.llm_config.model_name
        num_gpus = self.llm_config.num_gpus

        # Default parallel sizes
        tensor_parallel_size = num_gpus
        pipeline_parallel_size = 1

        # Adjust for specific model types or quantization
        if "8B" in model_name:
            tensor_parallel_size = 1
        
        # BitsAndBytes quantization requires specific settings
        if self.llm_config.quantization == 'bnb' or 'bnb' in model_name:
            quantization = 'bitsandbytes'
            load_format = 'bitsandbytes'
            tensor_parallel_size = 1
            pipeline_parallel_size = num_gpus
        else:
            quantization = self.llm_config.quantization
            load_format = self.llm_config.load_format

        return {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "seed": self.llm_config.seed,
            "dtype": "auto",
            "max_model_len": self.llm_config.max_model_len,
            "max_seq_len_to_capture": self.llm_config.max_model_len,
            "enable_prefix_caching": True,
            "enforce_eager": False,
            "gpu_memory_utilization": self.llm_config.gpu_memory_utilization,
            "quantization": quantization,
            "load_format": load_format,
            "trust_remote_code": True,
        }

    @staticmethod
    def _convert_messages_to_input_ids(
        messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer
    ) -> List[int]:
        """
        Applies the chat template and tokenizes a single conversation.

        :param messages: A list of messages in the conversation.
        :param tokenizer: The tokenizer to use.
        :return: A list of token IDs.
        """
        prompt = tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def infer(
        self, messages: List[TextChatMessage], **kwargs
    ) -> Tuple[str, Dict]:
        """
        Performs inference on a single conversation.

        :param messages: A list of chat messages for a single prompt.
        :param kwargs: Additional arguments, e.g., 'max_tokens'.
        :return: A tuple containing the response text and a metadata dictionary.
        """
        max_tokens = kwargs.get("max_tokens", 2048)
        logger.info("Calling VLLM offline for a single inference request.")
        
        # vLLM's generate method expects a batch, so we wrap the single prompt.
        prompt_ids = [self._convert_messages_to_input_ids(messages, self.tokenizer)]
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)

        vllm_output = self.client.generate(
            prompt_token_ids=prompt_ids, sampling_params=sampling_params
        )[0]
        
        response = vllm_output.outputs[0].text
        metadata = {
            "prompt_tokens": len(vllm_output.prompt_token_ids),
            "completion_tokens": len(vllm_output.outputs[0].token_ids),
        }
        return response, metadata

    def batch_infer(
        self, messages_list: List[List[TextChatMessage]], **kwargs
    ) -> Tuple[List[str], Dict]:
        """
        Performs batched inference on multiple conversations.

        :param messages_list: A list where each item is a conversation (list of messages).
        :param kwargs: Additional arguments, e.g., 'max_tokens' or 'json_template'.
        :return: A tuple containing a list of response texts and a metadata dictionary
                 summarizing the batch.
        """
        max_tokens = kwargs.get("max_tokens", 2048)
        json_template = kwargs.get("json_template")
        
        logger.info(f"Calling VLLM offline in batch mode for {len(messages_list)} requests.")

        guided = None
        if json_template:
            logger.info(f"Using guided JSON decoding with template: {json_template}")
            guided = GuidedDecodingRequest(guided_json=PROMPT_JSON_TEMPLATE[json_template])

        all_prompt_ids = [
            self._convert_messages_to_input_ids(messages, self.tokenizer)
            for messages in messages_list
        ]
        
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        vllm_outputs = self.client.generate(
            prompt_token_ids=all_prompt_ids,
            sampling_params=sampling_params,
            guided_options_request=guided,
        )

        all_responses = [output.outputs[0].text for output in vllm_outputs]
        metadata = {
            "prompt_tokens": sum(len(o.prompt_token_ids) for o in vllm_outputs),
            "completion_tokens": sum(len(o.outputs[0].token_ids) for o in vllm_outputs),
            "num_requests": len(messages_list),
        }
        return all_responses, metadata