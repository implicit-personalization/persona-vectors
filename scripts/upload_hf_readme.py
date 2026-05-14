#!/usr/bin/env python
"""Generate a concise dataset card from the vectors uploaded to Hugging Face."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from datasets import disable_progress_bars, load_dataset
from huggingface_hub import HfApi

BASELINE_PERSONA_ID = "baseline_assistant"
PARQUET_RE = re.compile(r"^(?P<config>[^/]+)/(?P<variant>.+?)-\d{5}-of-\d{5}\.parquet$")
disable_progress_bars()

KNOWN_LABELS = {
    "google__gemma-2-9b-it__answer_mean": ("train_test_split(n_train=50)", "all"),
    "google__gemma-3-27b-it__answer_mean": ("train_test_split(n_train=50)", "all"),
    "meta-llama__Llama-3.1-70B-Instruct__answer_mean": (
        "train_test_split(n_train=50)",
        "all",
    ),
    "meta-llama__Llama-3.1-405B-Instruct__answer_mean": (
        "all explicit questions",
        "explicit",
    ),
}


@dataclass(frozen=True)
class UploadedSplit:
    model: str
    config: str
    variant: str
    question_set: str
    qa_filter: str
    personas: int
    qa_range: str


def model_from_config(config: str) -> str:
    model_key, _, _mask_strategy = config.rpartition("__")
    return model_key.replace("__", "/")


def labels_for_config(
    config: str,
    current_config: str | None,
    current_labels: tuple[str, str] | None,
) -> tuple[str, str]:
    if config == current_config and current_labels is not None:
        return current_labels
    return KNOWN_LABELS.get(config, ("unknown", "unknown"))


def count_range(values: list[int]) -> str:
    unique = sorted(set(values))
    if not unique:
        return "unknown"
    return str(unique[0]) if len(unique) == 1 else f"{min(unique)}-{max(unique)}"


def uploaded_splits(repo: str) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in HfApi().list_repo_files(repo, repo_type="dataset"):
        match = PARQUET_RE.match(path)
        if match is not None:
            pairs.add((match["config"], match["variant"]))
    return sorted(pairs)


def summarize_split(
    repo: str,
    config: str,
    variant: str,
    current_config: str | None,
    current_labels: tuple[str, str] | None,
) -> UploadedSplit:
    dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{repo}/{config}/{variant}-*.parquet",
        split="train",
        columns=["persona_id", "sample_ids"],
    )
    sample_counts: list[int] = []
    for row in dataset:
        if row["persona_id"] == BASELINE_PERSONA_ID:
            continue
        sample_counts.append(len(row["sample_ids"] or []))

    question_set, qa_filter = labels_for_config(config, current_config, current_labels)
    return UploadedSplit(
        model=model_from_config(config),
        config=config,
        variant=variant,
        question_set=question_set,
        qa_filter=qa_filter,
        personas=len(sample_counts),
        qa_range=count_range(sample_counts),
    )


def grouped_rows(splits: list[UploadedSplit]) -> list[str]:
    groups: dict[tuple[str, str, str, str], list[UploadedSplit]] = {}
    for split in splits:
        key = (split.model, split.config, split.question_set, split.qa_filter)
        groups.setdefault(key, []).append(split)

    rows: list[str] = []
    for (model, config, question_set, qa_filter), group in sorted(groups.items()):
        variants = "; ".join(
            f"`{s.variant}`: {s.personas} personas, {s.qa_range} QA/persona"
            for s in sorted(group, key=lambda s: s.variant)
            if s.personas > 0
        )
        if variants:
            rows.append(
                f"| `{model}` | `{config}` | {variants} | {question_set} | `{qa_filter}` |"
            )
    return rows


def configs_yaml_block(pairs: list[tuple[str, str]]) -> list[str]:
    """Datasets-spec `configs:` YAML declaring one config per `{model}__{mask}`,
    one split per variant. Without this block the loader collapses everything
    to a single `default` config and `HFPersonaVectorStore` finds zero personas.
    """
    grouped: dict[str, list[str]] = {}
    for config, variant in pairs:
        grouped.setdefault(config, []).append(variant)

    lines = ["configs:"]
    for config in sorted(grouped):
        lines.append(f"- config_name: {config}")
        lines.append("  data_files:")
        for variant in sorted(grouped[config]):
            lines.append(f"  - split: {variant}")
            lines.append(f'    path: "{config}/{variant}-*.parquet"')
    return lines


def build_readme(
    repo: str,
    current_config: str | None = None,
    current_labels: tuple[str, str] | None = None,
) -> str:
    pairs = uploaded_splits(repo)
    splits = [
        summarize_split(repo, config, variant, current_config, current_labels)
        for config, variant in pairs
    ]
    rows = grouped_rows(splits)
    if not rows:
        raise RuntimeError(f"No persona vector parquet files found in {repo!r}")

    return "\n".join(
        [
            "---",
            "library_name: datasets",
            "tags:",
            "- persona-vectors",
            "- activation-vectors",
            *configs_yaml_block(pairs),
            "---",
            "",
            "# Persona Vectors",
            "",
            "Mean activation vectors extracted from synthetic persona prompts.",
            "This card is generated from the parquet files currently uploaded to the Hub.",
            "Counts exclude the `baseline_assistant` row.",
            "",
            "## Available Vectors",
            "",
            "| Model | Config | Variants | Question set | QA filter |",
            "| --- | --- | --- | --- | --- |",
            *rows,
            "",
            "## Columns",
            "",
            "- `persona_id`: stable persona identifier",
            "- `name`: persona display name",
            "- `sample_ids`: QA ids averaged into the vector",
            "- `vector`: `(num_layers, hidden_size)` float32 activation tensor",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--current-config",
        help="Config just pushed by the calling extraction script, e.g. google__gemma-2-9b-it__answer_mean.",
    )
    parser.add_argument(
        "--question-set",
        help="Question-set label for --current-config.",
    )
    parser.add_argument(
        "--qa-filter",
        help="QA filter label for --current-config.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    current_labels = None
    if args.current_config or args.question_set or args.qa_filter:
        if not (args.current_config and args.question_set and args.qa_filter):
            parser.error(
                "--current-config, --question-set, and --qa-filter must be passed together"
            )
        current_labels = (args.question_set, args.qa_filter)

    readme = build_readme(args.repo, args.current_config, current_labels)
    if args.dry_run:
        print(readme)
        return

    HfApi().upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    print(f"updated README.md -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
