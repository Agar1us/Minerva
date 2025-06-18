import asyncio
import importlib
import os
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Union

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    """
    Loads python files from ``<package_dir>/prompts/templates``.  
    Each file must expose a variable **prompt_template** containing either:

    1. a ``str`` / ``Template`` – rendered as a single string;  
    2. a chat history ``List[Dict[str, str]]`` with keys ``role`` and ``content``.
       In this case each ``content`` is converted to a ``Template`` and rendered
       individually.

    :param role_mapping: Remaps default roles (system/user/assistant) to those
                         required by a specific LLM provider.
    """

    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {"system": "system", "user": "user", "assistant": "assistant"}
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """
        Computes the template directory path and loads templates immediately.

        :return: ``None``.
        """
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)

        self.templates_dir = os.path.join(package_dir, "templates")
        self._load_templates()

    def _load_templates(self) -> None:
        """
        Imports every ``*.py`` in ``templates_dir`` and registers its
        ``prompt_template`` variable.

        :raises FileNotFoundError: If the directory is missing.
        :raises AttributeError:    If a module lacks *prompt_template*.
        :raises TypeError:         If *prompt_template* has an invalid format.
        """
        if not os.path.exists(self.templates_dir):
            msg = f"Templates directory '{self.templates_dir}' does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Loading templates from %s", self.templates_dir)

        for filename in os.listdir(self.templates_dir):
            if not (filename.endswith(".py") and filename != "__init__.py"):
                continue

            script_name = os.path.splitext(filename)[0]

            try:
                try:
                    module_name = f"src.hipporag.prompts.templates.{script_name}"
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    module_name = f".prompts.templates.{script_name}"
                    module = importlib.import_module(module_name, "hipporag")

                if not hasattr(module, "prompt_template"):
                    raise AttributeError(f"Module '{module_name}' lacks `prompt_template`.")

                prompt_template = module.prompt_template
                logger.debug("Loaded template from %s", module_name)

                if isinstance(prompt_template, Template):
                    self.templates[script_name] = prompt_template

                elif isinstance(prompt_template, str):
                    self.templates[script_name] = Template(prompt_template)

                elif isinstance(prompt_template, list) and all(
                    isinstance(item, dict) and {"role", "content"} <= item.keys()
                    for item in prompt_template
                ):
                    for item in prompt_template:
                        item["role"] = self.role_mapping.get(item["role"], item["role"])
                        item["content"] = (
                            item["content"]
                            if isinstance(item["content"], Template)
                            else Template(item["content"])
                        )
                    self.templates[script_name] = prompt_template

                else:
                    raise TypeError(
                        f"Invalid format in '{module_name}'. "
                        "Expected Template, str or chat-history list."
                    )

                logger.debug("Successfully registered template '%s'", script_name)

            except Exception as exc:
                logger.error("Failed to load template '%s': %s", script_name, exc)
                raise

    # ------------------------------------------------------------------ #
    # =========================== Public API =========================== #
    # ------------------------------------------------------------------ #
    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        Renders a template by substituting the supplied variables.

        :param name:   Template name (derived from filename).
        :param kwargs: Placeholder values for substitution.
        :return:       Rendered string *or* chat-history list.
        :raises ValueError: If a placeholder is missing.
        :raises KeyError:   If the template does not exist.
        """
        template = self.get_template(name)

        if isinstance(template, Template):
            try:
                result = template.substitute(**kwargs)
                logger.debug("Rendered '%s' with %s", name, kwargs)
                return result
            except KeyError as exc:
                raise ValueError(f"Missing variable for template '{name}': {exc}") from exc

        try:
            rendered = [
                {"role": item["role"], "content": item["content"].substitute(**kwargs)}
                for item in template
            ]
            logger.debug("Rendered chat template '%s' with %s", name, kwargs)
            return rendered
        except KeyError as exc:
            raise ValueError(f"Missing variable in chat template '{name}': {exc}") from exc

    def list_template_names(self) -> List[str]:
        """
        :return: List of all registered template names.
        """
        logger.info("Listing available prompt templates.")
        return list(self.templates.keys())

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        Retrieves the raw template object (without substitution).

        :param name: Template name.
        :return:     Template instance or chat-history list.
        :raises KeyError: If the template is missing.
        """
        if name not in self.templates:
            logger.error("Template '%s' not found.", name)
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]

    def print_template(self, name: str) -> None:
        """
        Pretty-prints a template’s content to stdout (for debugging).

        :param name: Template name.
        :raises KeyError: If the template is missing.
        """
        template = self.get_template(name)

        print(f"Template: {name}")
        if isinstance(template, Template):
            print(template.template)
        else:
            for item in template:
                print(f"Role: {item['role']}\nContent:\n{item['content'].template}\n---")

        logger.info("Printed template '%s'.", name)

    def is_template_name_valid(self, name: str) -> bool:
        """
        Checks whether a template name is registered.

        :param name: Template name.
        :return:     ``True`` if present, else ``False``.
        """
        return name in self.templates