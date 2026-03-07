import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def get_personas_path() -> Path:
    """Return the path to the personas JSONL file, from PERSONAS_PATH env or default."""
    return Path(os.environ.get("PERSONAS_PATH", "dataset_personas.jsonl"))


@dataclass
class QAPair:
    qid: str
    type: Literal["explicit", "implicit"]
    question: str
    answer: str
    difficulty: int  # 1 = easy, 2 = medium, 3 = hard

    def __repr__(self):
        return f"QAPair(qid={self.qid!r}, type={self.type!r}, difficulty={self.difficulty})"


@dataclass
class PersonaData:
    id: str
    persona: dict
    templated_prompt: str
    biography_md: str
    qa_pairs: list[QAPair]

    @property
    def name(self) -> str:
        return f"{self.persona['first_name']} {self.persona['last_name']}"

    def __repr__(self):
        return (
            f"PersonaData(id={self.id!r}, name={self.name!r}, "
            f"qa_pairs={len(self.qa_pairs)})"
        )


def load_personas(path: str | Path = "dataset_personas.jsonl") -> list[PersonaData]:
    """Load all personas from a JSONL file, one record per line."""
    personas = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            personas.append(
                PersonaData(
                    id=data["id"],
                    persona=data["persona"],
                    templated_prompt=data["templated_prompt"],
                    biography_md=data["biography_md"],
                    qa_pairs=[
                        QAPair(
                            qid=qp["qid"],
                            type=qp["type"],
                            question=qp["question"],
                            answer=qp["answer"],
                            difficulty=qp["difficulty"],
                        )
                        for qp in data.get("qa_pairs", [])
                    ],
                )
            )
    return personas


def get_qa_pairs(
    persona: PersonaData,
    type: Literal["explicit", "implicit"] | None = None,
    difficulty: int | None = None,
    as_text: bool = False,
) -> list[QAPair] | list[tuple[str, str]]:
    """Return qa_pairs filtered by type and/or difficulty.

    If as_text=True, returns (question, answer) tuples instead of QAPair objects.
    """
    pairs = persona.qa_pairs

    if type is not None:
        pairs = [p for p in pairs if p.type == type]

    if difficulty is not None:
        pairs = [p for p in pairs if p.difficulty == difficulty]

    if as_text:
        return [(p.question, p.answer) for p in pairs]

    return pairs
