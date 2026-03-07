import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from src.environment import get_hf_dataset_repo


@dataclass
class QAPair:
    id: str
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

    @property
    def name(self) -> str:
        return f"{self.persona['first_name']} {self.persona['last_name']}"

    def __repr__(self):
        return f"PersonaData(id={self.id!r}, name={self.name!r})"


def _resolve_path(
    filename: str,
    path: str | Path | None,
    from_hf: bool,
    hf_repo: str | None,
) -> Path:
    """Resolve a local path or download from HuggingFace."""
    if from_hf:
        repo = hf_repo or get_hf_dataset_repo()
        return Path(hf_hub_download(repo, filename, repo_type="dataset"))
    if path is None:
        raise ValueError("path is required when from_hf=False")
    return Path(path)


def load_personas(
    path: str | Path | None = None,
    from_hf: bool = False,
    hf_repo: str | None = None,
) -> list[PersonaData]:
    """Load all personas from a JSONL file or HuggingFace.

    Args:
        path: Local path to dataset_personas.jsonl. Required if from_hf=False.
        from_hf: If True, download from HuggingFace (cached via HF_HOME).
        hf_repo: HuggingFace dataset repo id.

    Returns:
        List of PersonaData objects.
    """
    resolved = _resolve_path("dataset_personas.jsonl", path, from_hf, hf_repo)
    personas = []
    with open(resolved, "r") as f:
        for line in f:
            data = json.loads(line)
            personas.append(
                PersonaData(
                    id=data["id"],
                    persona=data["persona"],
                    templated_prompt=data["templated_prompt"],
                    biography_md=data["biography_md"],
                )
            )
    return personas


def load_qa_pairs(
    path: str | Path | None = None,
    from_hf: bool = False,
    hf_repo: str | None = None,
) -> dict[str, list[QAPair]]:
    """Load QA pairs from a JSONL file or HuggingFace, grouped by persona id.

    Args:
        path: Local path to dataset_qa.jsonl. Required if from_hf=False.
        from_hf: If True, download from HuggingFace (cached via HF_HOME).
        hf_repo: HuggingFace dataset repo id.

    Returns:
        Dict mapping persona id -> list of QAPair objects.
    """
    resolved = _resolve_path("dataset_qa.jsonl", path, from_hf, hf_repo)
    qa_by_persona: dict[str, list[QAPair]] = {}
    with open(resolved, "r") as f:
        for line in f:
            data = json.loads(line)
            persona_id = data["id"]
            pair = QAPair(
                id=persona_id,
                qid=data["qid"],
                type=data["type"],
                question=data["question"],
                answer=data["answer"],
                difficulty=data["difficulty"],
            )
            qa_by_persona.setdefault(persona_id, []).append(pair)
    return qa_by_persona
