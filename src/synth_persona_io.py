import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from huggingface_hub import hf_hub_download


# NOTE: Think of adding @dataclass(slots=True)
# If we have memory problems might be something to look into more
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

    @property
    def name(self) -> str:
        return f"{self.persona['first_name']} {self.persona['last_name']}"

    def __repr__(self):
        return f"PersonaData(id={self.id!r}, name={self.name!r})"


class SynthPersonaDataset:
    """SynthPersona dataset loaded from HuggingFace."""

    def __init__(self, hf_repo: str = "implicit-personalization/synth-persona") -> None:
        # Download both files (HF Hub caches locally under HF_HOME so repeat runs are instant).
        personas_path = Path(
            hf_hub_download(hf_repo, "dataset_personas.jsonl", repo_type="dataset")
        )
        qa_path = Path(
            hf_hub_download(hf_repo, "dataset_qa.jsonl", repo_type="dataset")
        )

        with open(personas_path) as f:
            self._personas: list[PersonaData] = [
                PersonaData(
                    id=d["id"],
                    persona=d["persona"],
                    templated_prompt=d["templated_prompt"],
                    biography_md=d["biography_md"],
                )
                for d in (json.loads(line) for line in f)
            ]

        self._qa: dict[str, list[QAPair]] = defaultdict(list)
        with open(qa_path) as f:
            for line in f:
                d = json.loads(line)
                self._qa[d["id"]].append(
                    QAPair(
                        qid=d["qid"],
                        type=d["type"],
                        question=d["question"],
                        answer=d["answer"],
                        difficulty=d["difficulty"],
                    )
                )

    def __repr__(self) -> str:
        return f"SynthPersonaDataset(n_personas={len(self._personas)})"

    def __len__(self) -> int:
        return len(self._personas)

    def __iter__(self) -> Iterator[PersonaData]:
        return iter(self._personas)

    def __getitem__(self, idx: int) -> PersonaData:
        return self._personas[idx]

    def get_qa(
        self,
        persona_id: str,
        type: Literal["explicit", "implicit"] | None = None,
        difficulty: int | list[int] | None = None,
    ) -> list[QAPair]:
        """Return QA pairs for a persona, optionally filtered by type and/or difficulty.

        Args:
            persona_id: The persona id to look up.
            type: Keep only "explicit" or "implicit" pairs.
            difficulty: Keep only pairs at this level (1/2/3) or list of levels.
        """
        pairs = self._qa.get(persona_id, [])
        if type is not None:
            pairs = [p for p in pairs if p.type == type]
        if difficulty is not None:
            levels = {difficulty} if isinstance(difficulty, int) else set(difficulty)
            pairs = [p for p in pairs if p.difficulty in levels]
        return pairs

    def questions(
        self,
        persona_id: str,
        type: Literal["explicit", "implicit"] | None = None,
        difficulty: int | list[int] | None = None,
    ) -> list[str]:
        """Like get_qa but returns question strings only."""
        return [qa.question for qa in self.get_qa(persona_id, type, difficulty)]
