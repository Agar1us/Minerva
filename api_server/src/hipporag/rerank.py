import json
import re
import ast
import difflib
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional
from pydantic import TypeAdapter, BaseModel, Field
from .prompts.filter_default_prompt import best_dspy_prompt

DEFAULT_MAX_COMPLETION_TOKENS = 512
SIMILARITY_CUTOFF = 0.0
FIELD_HEADER_PATTERN = re.compile(r'\[\[ ## (\w+) ## \]\]')

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class LLMFactReranker:
    """
    A filtering system that uses language models to rerank facts based on query relevance.
    
    This class implements a DSPy-based filtering mechanism that processes candidate facts
    and reranks them according to their relevance to a given query using LLM inference.
    """

    def __init__(self, hipporag):
        """
        Initializes the DSPy filter with necessary configurations and templates.

        :param hipporag: HippoRAG instance providing global configuration and LLM model.
        """
        self.global_config = hipporag.global_config
        self.llm_model = hipporag.llm_model

        self._setup_templates()
        self._setup_generation_params()

    def _setup_templates(self) -> None:
        """
        Sets up input and output templates and creates the message template.
        """
        self.one_input_template = self._get_input_template()
        self.one_output_template = self._get_output_template()
        self.message_template = self._create_message_template()

    def _setup_generation_params(self) -> None:
        """
        Initializes default generation parameters for LLM inference.
        """
        self.default_gen_kwargs = {
            'max_completion_tokens': DEFAULT_MAX_COMPLETION_TOKENS
        }

    def _get_input_template(self) -> str:
        """
        Returns the input template for formatting user messages.

        :return: Formatted input template string.
        """
        return (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
            "Respond with the corresponding output fields, starting with the field "
            "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), "
            "and then ending with the marker for `[[ ## completed ## ]]`."
        )

    def _get_output_template(self) -> str:
        """
        Returns the output template for formatting assistant responses.

        :return: Formatted output template string.
        """
        return "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"

    def _create_message_template(self) -> List[Dict[str, str]]:
        """
        Creates the complete message template including system prompt and demonstrations.

        :return: List of message dictionaries containing role and content.
        """
        dspy_data = self._load_dspy_configuration()
        
        message_template = [
            {"role": "system", "content": dspy_data['prog']['system']}
        ]
        
        self._add_demonstrations(message_template, dspy_data['prog']['demos'])
        return message_template

    def _load_dspy_configuration(self) -> Dict[str, Any]:
        """
        Loads DSPy configuration from file or returns default configuration.

        :return: Dictionary containing DSPy configuration data.
        """
        dspy_file_path = self.global_config.rerank_dspy_file_path
        
        if dspy_file_path is not None:
            try:
                with open(dspy_file_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load DSPy file {dspy_file_path}: {e}")
                
        return best_dspy_prompt

    def _add_demonstrations(self, message_template: List[Dict[str, str]], demos: List[Dict[str, Any]]) -> None:
        """
        Adds demonstration examples to the message template.

        :param message_template: List to append demonstration messages to.
        :param demos: List of demonstration dictionaries.
        """
        for demo in demos:
            user_content = self.one_input_template.format(
                question=demo["question"],
                fact_before_filter=demo["fact_before_filter"]
            )
            assistant_content = self.one_output_template.format(
                fact_after_filter=demo["fact_after_filter"]
            )
            
            message_template.extend([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])

    def _parse_response_sections(self, response: str) -> List[Tuple[Optional[str], str]]:
        """
        Parses response into sections based on field headers.

        :param response: Raw response string from LLM.
        :return: List of tuples containing section name and content.
        """
        sections = [(None, [])]
        
        for line in response.splitlines():
            match = FIELD_HEADER_PATTERN.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        return [(k, "\n".join(v).strip()) for k, v in sections]

    def _parse_fact_value(self, value: str) -> Any:
        """
        Attempts to parse a string value into a Python object.

        :param value: String value to parse.
        :return: Parsed Python object or original string if parsing fails.
        """
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value

    def parse_filter(self, response: str) -> List[Any]:
        """
        Parses the LLM response to extract filtered facts.

        :param response: Raw response string from the language model.
        :return: List of parsed and validated facts.
        """
        sections = self._parse_response_sections(response)
        parsed_facts = []
        
        for section_name, section_content in sections:
            if section_name == "fact_after_filter":
                try:
                    parsed_value = self._parse_fact_value(section_content)
                    validated_fact = TypeAdapter(Fact).validate_python(parsed_value).fact
                    parsed_facts = validated_fact
                except Exception as e:
                    print(f"Error parsing field {section_name}: {e}.\n"
                          f"Attempted to parse:\n```\n{section_content}\n```")

        return parsed_facts

    def _create_llm_messages(self, question: str, fact_before_filter: str) -> List[Dict[str, str]]:
        """
        Creates the complete message list for LLM inference.

        :param question: The input question.
        :param fact_before_filter: JSON string of facts to filter.
        :return: List of message dictionaries.
        """
        messages = deepcopy(self.message_template)
        user_message = {
            "role": "user",
            "content": self.one_input_template.format(
                question=question,
                fact_before_filter=fact_before_filter
            )
        }
        messages.append(user_message)
        return messages

    async def llm_call(self, question: str, fact_before_filter: str) -> str:
        """
        Makes an inference call to the language model.

        :param question: The input question to process.
        :param fact_before_filter: JSON string containing facts to filter.
        :return: Response string from the language model.
        """
        messages = self._create_llm_messages(question, fact_before_filter)
        
        response = await self.llm_model.async_infer(
            messages=messages,
            **self.default_gen_kwargs
        )

        return response[0] if len(response) > 1 else response

    def _find_matching_fact_indices(self, generated_facts: List[Any], candidate_items: List[Tuple]) -> List[int]:
        """
        Finds indices of candidate items that match the generated facts.

        :param generated_facts: List of facts generated by the LLM.
        :param candidate_items: List of candidate fact tuples.
        :return: List of indices corresponding to matched facts.
        """
        result_indices = []
        candidate_strings = [str(item) for item in candidate_items]
        
        for generated_fact in generated_facts:
            try:
                closest_matches = difflib.get_close_matches(
                    str(generated_fact), 
                    candidate_strings, 
                    n=1, 
                    cutoff=SIMILARITY_CUTOFF
                )
                
                if closest_matches:
                    closest_match = closest_matches[0]
                    original_fact = eval(closest_match)
                    result_indices.append(candidate_items.index(original_fact))
                    
            except (ValueError, SyntaxError) as e:
                print(f"Error finding matching fact index: {e}")
                continue

        return result_indices

    async def rerank(self, 
               query: str, 
               candidate_items: List[Tuple], 
               candidate_indices: List[int],
               len_after_rerank: Optional[int] = None) -> Tuple[List[int], List[Tuple], Dict[str, Any]]:
        """
        Reranks candidate items based on query relevance using LLM filtering.

        :param query: The input query for relevance assessment.
        :param candidate_items: List of candidate fact tuples to rerank.
        :param candidate_indices: List of original indices for candidate items.
        :param len_after_rerank: Maximum number of items to return after reranking.
        :return: Tuple containing reranked indices, items, and metadata dictionary.
        """
        fact_before_filter = {"fact": [list(item) for item in candidate_items]}
        
        try:
            response = await self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print(f'Exception during LLM call or parsing: {e}')
            generated_facts = []
        
        result_indices = self._find_matching_fact_indices(generated_facts, candidate_items)
        
        if len_after_rerank is not None:
            result_indices = result_indices[:len_after_rerank]
        
        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        
        return sorted_candidate_indices, sorted_candidate_items, {'confidence': None}

    async def __call__(self, *args, **kwargs):
        """
        Makes the instance callable, delegating to the rerank method.

        :param args: Positional arguments passed to rerank.
        :param kwargs: Keyword arguments passed to rerank.
        :return: Result from rerank method.
        """
        return await self.rerank(*args, **kwargs)
