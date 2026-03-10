# Dataset Loading

Each dataset has its own file in `src/` (e.g. `synth_persona_io.py`). Construction
downloads files from HuggingFace (cached locally via `HF_HOME`) and loads everything
eagerly in `__init__`.

`SynthPersonaDataset` is the reference implementation. New datasets should follow
a similar structure.

---

## SynthPersona

```python
from src.synth_persona_io import SynthPersonaDataset

dataset = SynthPersonaDataset()                          # default repo
dataset = MyDataset()                                    # custom repo
```

Iterate and index over personas:

```python
len(dataset)          # number of personas
dataset[0]            # PersonaData
for persona in dataset: ...
```

Query QA pairs — all filters are optional and combinable:

```python
persona = dataset[0]

# This retuns the object
dataset.get_qa(persona.id)                                      # all pairs
dataset.get_qa(persona.id, type="explicit", difficulty=[2, 3])  # combined

# This returns the strings directlly which is a useful utility
dataset.questions(persona.id, type="implicit")
```

---

## Adding a new dataset

Create `src/<name>_io.py`. The pattern is:

1. Define `@dataclass` records for your data.
2. Write a plain class with `__init__` that downloads files via
   `hf_hub_download`, loads lightweight records
3. Expose `__len__`, `__iter__`, `__getitem__`, and other relevant things similar to what I have done

### Example: PersonaGuess

> AI Generated example to showcase possible implementation of another dataset

```python
@dataclass
class Turn:
    round: int
    asker: Literal["A", "B"]
    question: str
    answer: str


@dataclass
class GameRecord:
    game_id: str
    persona_a_id: str
    persona_b_id: str
    turns: list[Turn]


class PersonaGuessDataset:
    DEFAULT_REPO = "implicit-personalization/persona-guess"

    def __init__(self, hf_repo: str = DEFAULT_REPO) -> None:
        path = Path(hf_hub_download(hf_repo, "games.jsonl", repo_type="dataset"))
        with open(path) as f:
            self._games: list[GameRecord] = [
                GameRecord(
                    game_id=d["game_id"],
                    persona_a_id=d["persona_a_id"],
                    persona_b_id=d["persona_b_id"],
                    turns=[Turn(**t) for t in d["turns"]],
                )
                for d in (json.loads(line) for line in f)
            ]

    def __len__(self) -> int:
        return len(self._games)

    def __iter__(self) -> Iterator[GameRecord]:
        return iter(self._games)

    def __getitem__(self, idx: int) -> GameRecord:
        return self._games[idx]

    def get_qa(self, game_id: str, player: Literal["A", "B"] | None = None) -> list[Turn]:
        game = next(g for g in self._games if g.game_id == game_id)
        return [t for t in game.turns if player is None or t.asker == player]

    def questions(self, game_id: str, player: Literal["A", "B"] | None = None) -> list[str]:
        return [t.question for t in self.get_qa(game_id, player)]
```

Usage:

```python
from src.persona_guess_io import PersonaGuessDataset

games = PersonaGuessDataset()
game = games[0]
questions = games.questions(game.game_id, player="A")
```
