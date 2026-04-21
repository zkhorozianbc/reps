"""
Template loader for REPS.

Templates live as .txt files under reps/prompt_templates/ and can be overridden
per-run by pointing a custom directory at TemplateManager.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TemplateManager:
    """Loads .txt templates and fragments.json with optional custom override directory."""

    def __init__(self, custom_template_dir: Optional[str] = None):
        self.default_dir = Path(__file__).parent / "prompt_templates"
        self.custom_dir = Path(custom_template_dir) if custom_template_dir else None

        self.templates: dict = {}
        self.fragments: dict = {}

        self._load_from_directory(self.default_dir)

        if self.custom_dir:
            if self.custom_dir.exists():
                self._load_from_directory(self.custom_dir)
            else:
                logger.warning("Custom template directory does not exist, using default prompt.")

    def _load_from_directory(self, directory: Path) -> None:
        if not directory.exists():
            return

        for txt_file in directory.glob("*.txt"):
            with open(txt_file, "r") as f:
                self.templates[txt_file.stem] = f.read()

        fragments_file = directory / "fragments.json"
        if fragments_file.exists():
            with open(fragments_file, "r") as f:
                self.fragments.update(json.load(f))

    def get_template(self, name: str) -> str:
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def get_fragment(self, name: str, **kwargs) -> str:
        if name not in self.fragments:
            return f"[Missing fragment: {name}]"
        try:
            return self.fragments[name].format(**kwargs)
        except KeyError as e:
            return f"[Fragment formatting error: {e}]"

    def add_template(self, template_name: str, template: str) -> None:
        self.templates[template_name] = template

    def add_fragment(self, fragment_name: str, fragment: str) -> None:
        self.fragments[fragment_name] = fragment
