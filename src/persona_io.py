import json
import os
from pathlib import Path
from typing import TypedDict


def get_personas_path() -> Path:
    """Get the path to the personas dataset file."""
    return Path(os.environ.get("PERSONAS_PATH", "dataset_personas.jsonl"))


class PersonaData(TypedDict):
    id: str
    persona: dict
    templated_prompt: str
    biography_md: str


def load_personas(path: str | Path = "dataset_personas.jsonl") -> list[PersonaData]:
    """Load personas from a JSONL file.

    Args:
        path: Path to the JSONL file containing personas.

    Returns:
        List of persona data dicts, each containing:
        - id: Unique identifier
        - persona: Dict of persona attributes
        - templated_prompt: Short prompt version
        - biography_md: Long narrative biography
    """
    personas = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            personas.append(
                {
                    "id": data["id"],
                    "persona": data["persona"],
                    "templated_prompt": data["templated_prompt"],
                    "biography_md": data["biography_md"],
                }
            )
    return personas
