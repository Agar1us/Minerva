import json
import re
from string import Template
from typing import List, Optional, Sequence, TypedDict, Union

# third-party
import tiktoken

# --------------------------------------------------------------------------- #
# ============================ Type aliases ================================= #
# --------------------------------------------------------------------------- #
class TextChatMessage(TypedDict):
    """
    Representation of a single chat message.

    :param role:    Either ``"system"``, ``"user"`` or ``"assistant"``.
    :param content: Raw text (or a string.Template) of the message body.
    """

    role: str
    content: Union[str, Template]


# --------------------------------------------------------------------------- #
# ========================= Placeholder conversion ========================== #
# --------------------------------------------------------------------------- #
def convert_format_to_template(
    original_string: str,
    placeholder_mapping: Optional[dict[str, str]] = None,
    static_values: Optional[dict[str, str]] = None,
) -> str:
    """
    Converts ``str.format`` placeholders (``{name}``) to ``string.Template``
    placeholders (``${name}``).  Optionally renames or hard-codes placeholders.

    :param original_string:     Source string with ``{…}`` placeholders.
    :param placeholder_mapping: Map ``old_name -> new_name`` for renaming.
    :param static_values:       Map ``name -> static_value``; such slots are
                                replaced by the provided literal.
    :return:                    Converted string ready for ``Template`` usage.
    """
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}

    pattern = re.compile(r"\{(\w+)\}")

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in static_values:
            return str(static_values[key])
        return f"${{{placeholder_mapping.get(key, key)}}}"

    return pattern.sub(_replace, original_string)


# --------------------------------------------------------------------------- #
# ========================= Safe Unicode decode ============================= #
# --------------------------------------------------------------------------- #
def safe_unicode_decode(content: Union[bytes, str]) -> str:
    """
    Decodes `\\uXXXX` escape sequences regardless of whether *content* is
    already a string or raw bytes.

    :param content: Byte string or normal string potentially containing escapes.
    :return:        Decoded, human-readable string.
    :raises TypeError: If *content* is of an unsupported type.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    elif not isinstance(content, str):
        raise TypeError("Input must be bytes or str.")

    pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    def _decode(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 16))

    return pattern.sub(_decode, content)


# --------------------------------------------------------------------------- #
# ======================== Fix broken JSON ================================== #
# --------------------------------------------------------------------------- #
def fix_broken_generated_json(json_str: str) -> str:
    """
    Attempts to repair a *truncated* JSON object produced by an LLM.  
    Strategy:

    1. If ``json.loads`` succeeds — return as-is.  
    2. Drop everything after the last comma.  
    3. Find unclosed ``{`` / ``[`` and append matching closing braces.

    :param json_str: Raw possibly-broken JSON string.
    :return:         A string that *should* be parseable by ``json.loads``.
    """

    def _find_unclosed(text: str) -> List[str]:
        stack: List[str] = []
        inside_string = escape_next = False

        for ch in text:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    inside_string = False
            else:
                if ch == '"':
                    inside_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                        stack.pop()
        return stack

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    last_comma = json_str.rfind(",")
    if last_comma != -1:
        json_str = json_str[:last_comma]

    closing_map = {"{": "}", "[": "]"}
    for opener in reversed(_find_unclosed(json_str)):
        json_str += closing_map[opener]

    return json_str


# --------------------------------------------------------------------------- #
# ========================= Triple post-filter ============================== #
# --------------------------------------------------------------------------- #
def filter_invalid_triples(triples: Sequence[Sequence[str]]) -> List[List[str]]:
    """
    Removes invalid (size != 3) or duplicate triples.

    :param triples: Iterable ``[(s, p, o), …]`` (any element convertible to str).
    :return:        Unique list preserving the original order.
    """
    seen: set[tuple[str, str, str]] = set()
    valid: List[List[str]] = []

    for triple in triples:
        if len(triple) != 3:
            continue
        triplet = tuple(str(x) for x in triple)
        if triplet not in seen:
            seen.add(triplet)
            valid.append(list(triplet))

    return valid


# --------------------------------------------------------------------------- #
# ================= JSON schema for guided decoding ========================= #
# --------------------------------------------------------------------------- #
PROMPT_JSON_TEMPLATE: dict[str, dict] = {
    "ner": {
        "type": "object",
        "properties": {
            "named_entities": {"type": "array", "items": {"type": "string"}, "minItems": 0}
        },
        "required": ["named_entities"],
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0,
            }
        },
        "required": ["triples"],
    },
    "fact": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0,
            }
        },
        "required": ["fact"],
    },
    "json": {"type": "object"},
    "qa_cot": {
        "type": "object",
        "required": ["Thought", "Answer"],
        "properties": {
            "Thought": {"type": "string", "minLength": 1, "maxLength": 2000},
            "Answer": {"type": "string", "minLength": 1, "maxLength": 200},
        },
    },
}

# --------------------------------------------------------------------------- #
# ======================== Token counting via tiktoken ====================== #
# --------------------------------------------------------------------------- #
def num_tokens_by_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Counts tokens in *text* using the same rules as OpenAI’s ``tiktoken`` lib.

    :param text:  Input string.
    :param model: Reference model whose encoding rules will be applied.
    :return:      Token count.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))