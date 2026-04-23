#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from persona_data.prompts import mc_correct_letter
from persona_data.synth_persona import SynthPersonaDataset

from persona_vectors.steering import _shared_item_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether item-conditioned oracle steering is driven by "
            "answer-letter leakage from response-token activations."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Item-conditioned oracle run root containing alpha_* outputs.",
    )
    parser.add_argument("--alpha-dir", default="alpha_8p0")
    parser.add_argument("--qa-type", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--questions-per-persona", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    alpha_root = args.run_root / args.alpha_dir
    summary_payload = json.loads((alpha_root / "summary.json").read_text())
    cross_source_for = summary_payload["metadata"]["cross_source_for"]

    dataset = SynthPersonaDataset()
    gold_by_persona_item: dict[tuple[str, str], str] = {}
    for persona in dataset:
        qa_pairs = [
            qa
            for qa in dataset.get_qa(persona.id)
            if qa.type == args.qa_type and qa.answer_format == "choice"
        ][: args.questions_per_persona]
        for qa in qa_pairs:
            item_key = _shared_item_key(
                qid=qa.qid,
                question=qa.question,
                persona_id=persona.id,
            )
            gold_by_persona_item[(persona.id, item_key)] = mc_correct_letter(qa)

    rows = [json.loads(line) for line in (alpha_root / "per_example.jsonl").open()]
    bare_by_key = {
        (row["persona_id"], row["qid"]): row
        for row in rows
        if row["condition"] == "bare"
    }

    cross_rows: list[dict] = []
    for row in rows:
        if row["condition"] != "steered_cross_item":
            continue
        item_key = _shared_item_key(
            qid=row["qid"],
            question=row["question"],
            persona_id=row["persona_id"],
        )
        source_id = cross_source_for[row["persona_id"]]
        source_gold = gold_by_persona_item.get((source_id, item_key))
        bare = bare_by_key[(row["persona_id"], row["qid"])]
        cross_rows.append(
            {
                "persona_id": row["persona_id"],
                "persona_name": row["persona_name"],
                "qid": row["qid"],
                "item_key": item_key,
                "target_gold": row["gold_letter"],
                "cross_source_id": source_id,
                "cross_source_gold": source_gold,
                "same_gold_as_cross_source": source_gold == row["gold_letter"],
                "bare_gold_prob": bare["gold_prob"],
                "steered_gold_prob": row["gold_prob"],
                "delta_gold_prob": row["gold_prob"] - bare["gold_prob"],
                "bare_correct": bare["correct"],
                "steered_correct": row["correct"],
            }
        )

    by_same_gold: dict[str, list[dict]] = defaultdict(list)
    for row in cross_rows:
        by_same_gold[str(row["same_gold_as_cross_source"])].append(row)

    cross_summary = {}
    for key, group in sorted(by_same_gold.items()):
        deltas = [row["delta_gold_prob"] for row in group]
        cross_summary[key] = {
            "n": len(group),
            "mean_delta_gold_prob": mean(deltas),
            "positive_delta_count": sum(delta > 0 for delta in deltas),
            "flip_to_gold": sum(
                (not row["bare_correct"]) and row["steered_correct"] for row in group
            ),
            "flip_away_from_gold": sum(
                row["bare_correct"] and (not row["steered_correct"]) for row in group
            ),
        }

    own_rows: list[dict] = []
    for row in rows:
        if row["condition"] != "steered_own_item":
            continue
        bare = bare_by_key[(row["persona_id"], row["qid"])]
        own_rows.append(
            {
                "persona_id": row["persona_id"],
                "persona_name": row["persona_name"],
                "qid": row["qid"],
                "target_gold": row["gold_letter"],
                "bare_gold_prob": bare["gold_prob"],
                "steered_gold_prob": row["gold_prob"],
                "delta_gold_prob": row["gold_prob"] - bare["gold_prob"],
                "bare_correct": bare["correct"],
                "steered_correct": row["correct"],
            }
        )

    by_target_gold: dict[str, list[dict]] = defaultdict(list)
    for row in own_rows:
        by_target_gold[row["target_gold"]].append(row)
    own_summary = {}
    for letter, group in sorted(by_target_gold.items()):
        deltas = [row["delta_gold_prob"] for row in group]
        own_summary[letter] = {
            "n": len(group),
            "mean_delta_gold_prob": mean(deltas),
            "positive_delta_count": sum(delta > 0 for delta in deltas),
        }

    output = {
        "metadata": {
            "run_root": str(args.run_root),
            "alpha_dir": args.alpha_dir,
            "qa_type": args.qa_type,
            "questions_per_persona": args.questions_per_persona,
        },
        "cross_source_same_gold_summary": cross_summary,
        "own_target_gold_letter_summary": own_summary,
        "interpretation": (
            "If cross-source vectors lift target gold probability mainly when "
            "source and target share the same gold letter, the item-conditioned "
            "oracle is likely dominated by answer-letter leakage from response-token activations."
        ),
    }
    (alpha_root / "answer_letter_leakage_analysis.json").write_text(
        json.dumps(output, indent=2)
    )
    write_csv(alpha_root / "answer_letter_leakage_cross_rows.csv", cross_rows)
    write_csv(alpha_root / "answer_letter_leakage_own_rows.csv", own_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
