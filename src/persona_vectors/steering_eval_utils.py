from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import ActivationStore
from persona_vectors.eval import ChoiceEvalResult

console = Console()
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory")
TRANSIENT_REMOTE_MARKERS = (
    "RemoteError",
    "WriteTimeout",
    "ReadTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "timed out",
    "Connection reset by peer",
    "Server disconnected",
    "status ERROR",
    "Job failed",
    "job failed",
    "503",
    "502",
    "504",
)


class NonRetryableRemoteError(RuntimeError):
    """Remote error that should be recorded, not retried unchanged."""


class NonRetryableRemoteOOM(NonRetryableRemoteError):
    """A single remote prompt exceeded the remote job memory cap."""


def select_qa_pairs(dataset, persona_id: str, qa_type: str, limit: int):
    qa_pairs = [
        qa
        for qa in dataset.get_qa(persona_id)
        if qa.type == qa_type and qa.answer_format == "choice"
    ]
    return qa_pairs[:limit]


def is_oom_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in OOM_MARKERS)


def run_with_oom_retry(fn, *, label: str, retries: int = 3, sleep_seconds: int = 20):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_oom_error(exc) or attempt == retries:
                raise
            console.print(
                f"[yellow]{label} hit remote OOM on attempt {attempt}/{retries}; "
                f"sleeping {sleep_seconds}s and retrying[/]"
            )
            time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def is_transient_remote_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in TRANSIENT_REMOTE_MARKERS)


def run_with_remote_retry(
    fn,
    *,
    label: str,
    retries: int = 4,
    sleep_seconds: int = 8,
):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, NonRetryableRemoteError):
                retryable = False
                reason = "non-retryable remote error"
            elif is_oom_error(exc):
                retryable = True
                reason = "remote OOM"
            elif is_transient_remote_error(exc):
                retryable = True
                reason = "transient remote error"
            else:
                retryable = False
                reason = "non-retryable error"

            if not retryable or attempt == retries:
                raise

            wait_s = sleep_seconds * attempt
            console.print(
                f"[yellow]{label} hit {reason} on attempt {attempt}/{retries}; "
                f"sleeping {wait_s}s and retrying[/]"
            )
            time.sleep(wait_s)

    assert last_exc is not None
    raise last_exc


def load_existing_rows(path: Path) -> list[ChoiceEvalResult]:
    if not path.exists():
        return []

    rows: list[ChoiceEvalResult] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(ChoiceEvalResult(**json.loads(line)))
    return rows


def cached_variant_matches(
    store: ActivationStore,
    prompt_variant: str,
    persona_id: str,
    expected_qids: list[str],
    *,
    expected_prompt_contract_version: str | None = None,
) -> bool:
    try:
        _, qids, _ = store.load_records(prompt_variant, persona_id)
    except FileNotFoundError:
        return False

    if expected_prompt_contract_version is not None:
        metadata = store.load_metadata(prompt_variant, persona_id)
        if metadata.get("prompt_contract_version") != expected_prompt_contract_version:
            return False

    return qids == expected_qids


def summarize(rows: list[ChoiceEvalResult]) -> dict[str, dict[str, float | int]]:
    by_condition: dict[str, list[ChoiceEvalResult]] = defaultdict(list)
    for row in rows:
        by_condition[row.condition].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for condition, cond_rows in by_condition.items():
        n = len(cond_rows)
        summary[condition] = {
            "n_examples": n,
            "accuracy": sum(int(r.correct) for r in cond_rows) / n if n else 0.0,
            "mean_gold_prob": sum(r.gold_prob for r in cond_rows) / n if n else 0.0,
            "mean_gold_logprob": (
                sum(r.gold_logprob for r in cond_rows) / n if n else 0.0
            ),
            "mean_margin_vs_best_other": (
                sum(r.margin_vs_best_other for r in cond_rows) / n if n else 0.0
            ),
        }

    grouped: dict[tuple[str, str], dict[str, ChoiceEvalResult]] = defaultdict(dict)
    for row in rows:
        grouped[(row.persona_id, row.qid)][row.condition] = row

    for condition in sorted(by_condition):
        if condition == "bare":
            continue
        deltas: list[float] = []
        flip_to_gold = 0
        flip_away_from_gold = 0
        any_flip = 0
        comparable = 0
        for pair in grouped.values():
            if "bare" not in pair or condition not in pair:
                continue
            comparable += 1
            bare = pair["bare"]
            other = pair[condition]
            deltas.append(other.gold_prob - bare.gold_prob)
            if other.predicted_letter != bare.predicted_letter:
                any_flip += 1
            if (not bare.correct) and other.correct:
                flip_to_gold += 1
            if bare.correct and (not other.correct):
                flip_away_from_gold += 1

        if comparable:
            summary[f"{condition}_vs_bare"] = {
                "n_examples": comparable,
                "mean_delta_gold_prob": sum(deltas) / comparable,
                "flip_to_gold": flip_to_gold,
                "flip_away_from_gold": flip_away_from_gold,
                "any_flip": any_flip,
            }

    return summary


def write_outputs(
    out_dir: Path,
    rows: list[ChoiceEvalResult],
    summary: dict[str, dict[str, float | int]],
    metadata: dict,
    failures: list[dict] | None = None,
    *,
    jsonl_name: str = "per_example.jsonl",
    csv_name: str = "per_example.csv",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    row_dicts = [row.to_dict() for row in rows]

    with (out_dir / jsonl_name).open("w") as f:
        for row in row_dicts:
            f.write(json.dumps(row) + "\n")

    with (out_dir / "summary.json").open("w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)

    if failures is not None:
        with (out_dir / "failures.json").open("w") as f:
            json.dump(failures, f, indent=2)

    if row_dicts:
        fieldnames = list(row_dicts[0].keys())
        with (out_dir / csv_name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_dicts)


def render_summary(summary: dict[str, dict[str, float | int]], *, title: str) -> None:
    table = Table(title=title)
    table.add_column("Condition", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")

    for condition, metrics in summary.items():
        first = True
        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            table.add_row(condition if first else "", key, value_str)
            first = False

    console.print(table)
