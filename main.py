#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.activation_io import (
    load_contrastive_vectors,
    save_contrastive_vectors,
    save_per_prompt_summaries,
    save_persona_representations,
)
from src.activations import extract_activation_summaries
from src.environment import load_env, set_seed
from src.persona_io import (
    get_neutral_prompts_path,
    get_personas_path,
    load_neutral_prompts,
    load_personas,
)
from src.plots import pca_project_personas, plot_layer_similarity, save_projection_artifact
from src.prompt_format import format_messages


@dataclass
class ExtractConfig:
    model: str
    input_path: str
    neutral_prompts_path: str
    output_dir: str
    baseline_system_prompt: str
    remote: bool
    max_new_tokens: int
    do_sample: bool
    seed: int
    persona_limit: int | None


@dataclass
class AnalyzeConfig:
    activations_path: str
    input_path: str
    model_name: str
    output_dir: str
    prompt_formats: list[str]


def _load_model(model_name: str, remote: bool):
    dtype = torch.bfloat16
    if remote:
        print(f"Loading model remotely: {model_name}")
        return LanguageModel(model_name)
    print(f"Loading model locally: {model_name}")
    return LanguageModel(model_name, dtype=dtype, device_map="auto")


def _generate_response_and_mask(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    remote: bool,
) -> tuple[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt, _ = format_messages(messages, tokenizer)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = prompt_ids.shape[1]

    with model.generate(
        prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, remote=remote
    ) as tracer:
        result = tracer.result.save()

    response_text = str(tokenizer.decode(result[0][prompt_length:], skip_special_tokens=True))
    messages_full = messages + [{"role": "assistant", "content": response_text}]
    full_text, response_start_idx = format_messages(messages_full, tokenizer)
    token_mask = torch.arange(len(tokenizer(full_text).input_ids)) >= response_start_idx
    return full_text, token_mask


def _run_summary_extraction_for_context(
    model,
    tokenizer,
    system_prompt: str,
    neutral_prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    remote: bool,
) -> dict[str, torch.Tensor]:
    full_texts: list[str] = []
    token_masks: list[torch.Tensor] = []
    for neutral_prompt in tqdm(neutral_prompts, desc="Neutral prompts", leave=False):
        full_text, token_mask = _generate_response_and_mask(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=neutral_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            remote=remote,
        )
        full_texts.append(full_text)
        token_masks.append(token_mask)

    return extract_activation_summaries(
        model=model,
        full_texts=full_texts,
        token_masks=token_masks,
        remote=remote,
    )


def extract_activations(cfg: ExtractConfig) -> None:
    load_env()
    torch.set_grad_enabled(False)
    set_seed(cfg.seed)

    personas = load_personas(cfg.input_path)
    if cfg.persona_limit is not None:
        personas = personas[: cfg.persona_limit]
    neutral_prompts = load_neutral_prompts(cfg.neutral_prompts_path)

    model = _load_model(cfg.model, cfg.remote)
    tokenizer = model.tokenizer
    output_root = Path(cfg.output_dir)

    # Baselines are reused across personas per format.
    baseline_representations: dict[str, dict[str, torch.Tensor]] = {}

    variant_to_field = {
        "templated": "templated_prompt",
        "biography": "biography_md",
    }

    for prompt_variant in variant_to_field:
        print(f"\nExtracting baseline for variant={prompt_variant}")
        baseline_summary = _run_summary_extraction_for_context(
            model=model,
            tokenizer=tokenizer,
            system_prompt=cfg.baseline_system_prompt,
            neutral_prompts=neutral_prompts,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            remote=cfg.remote,
        )
        baseline_response_mean = baseline_summary["response_mean"].mean(dim=0)
        baseline_last_prompt = baseline_summary["last_prompt_token"].mean(dim=0)
        baseline_representations[prompt_variant] = {
            "response_mean": baseline_response_mean,
            "last_prompt_token": baseline_last_prompt,
        }
        save_per_prompt_summaries(
            root_dir=output_root,
            model_name=cfg.model,
            prompt_variant=prompt_variant,
            persona_id="__baseline__",
            neutral_prompts=neutral_prompts,
            response_mean_per_prompt=baseline_summary["response_mean"],
            last_prompt_per_prompt=baseline_summary["last_prompt_token"],
            metadata={
                "source": "baseline",
                "baseline_system_prompt": cfg.baseline_system_prompt,
            },
        )
        save_persona_representations(
            root_dir=output_root,
            model_name=cfg.model,
            prompt_variant=prompt_variant,
            persona_id="__baseline__",
            persona_response_mean=baseline_response_mean,
            persona_last_prompt=baseline_last_prompt,
            metadata={
                "source": "baseline",
                "baseline_system_prompt": cfg.baseline_system_prompt,
            },
        )

    for persona in tqdm(personas, desc="Personas"):
        for prompt_variant, field_name in variant_to_field.items():
            persona_context = getattr(persona, field_name)
            print(f"\nExtracting persona={persona.id} variant={prompt_variant}")
            summary = _run_summary_extraction_for_context(
                model=model,
                tokenizer=tokenizer,
                system_prompt=persona_context,
                neutral_prompts=neutral_prompts,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                remote=cfg.remote,
            )

            response_mean_per_prompt = summary["response_mean"]
            last_prompt_per_prompt = summary["last_prompt_token"]
            persona_response_mean = response_mean_per_prompt.mean(dim=0)
            persona_last_prompt = last_prompt_per_prompt.mean(dim=0)

            baseline = baseline_representations[prompt_variant]
            contrastive_response = persona_response_mean - baseline["response_mean"]
            contrastive_last = persona_last_prompt - baseline["last_prompt_token"]

            common_metadata = {
                "source": "persona",
                "persona_id": persona.id,
                "prompt_variant": prompt_variant,
                "neutral_prompts_path": cfg.neutral_prompts_path,
                "n_prompts": len(neutral_prompts),
                "model_name": cfg.model,
                "remote": cfg.remote,
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": cfg.do_sample,
                "seed": cfg.seed,
            }
            save_per_prompt_summaries(
                root_dir=output_root,
                model_name=cfg.model,
                prompt_variant=prompt_variant,
                persona_id=persona.id,
                neutral_prompts=neutral_prompts,
                response_mean_per_prompt=response_mean_per_prompt,
                last_prompt_per_prompt=last_prompt_per_prompt,
                metadata=common_metadata,
            )
            save_persona_representations(
                root_dir=output_root,
                model_name=cfg.model,
                prompt_variant=prompt_variant,
                persona_id=persona.id,
                persona_response_mean=persona_response_mean,
                persona_last_prompt=persona_last_prompt,
                metadata=common_metadata,
            )
            save_contrastive_vectors(
                root_dir=output_root,
                model_name=cfg.model,
                prompt_variant=prompt_variant,
                persona_id=persona.id,
                contrastive_response_mean=contrastive_response,
                contrastive_last_prompt=contrastive_last,
                metadata=common_metadata,
            )


def analyze_activations(cfg: AnalyzeConfig) -> None:
    load_env()
    # Reserved for future output routing; plots currently use ARTIFACTS_DIR/plots.
    _ = cfg.output_dir
    personas = load_personas(cfg.input_path)

    required_formats = set(cfg.prompt_formats)
    if required_formats != {"biography", "templated"}:
        raise ValueError("analyze currently expects prompt formats: biography,templated")

    biography_vectors: dict[str, torch.Tensor] = {}
    templated_vectors: dict[str, torch.Tensor] = {}

    for persona in personas:
        bio_response, _, _ = load_contrastive_vectors(
            root_dir=cfg.activations_path,
            model_name=cfg.model_name,
            prompt_variant="biography",
            persona_id=persona.id,
        )
        templ_response, _, _ = load_contrastive_vectors(
            root_dir=cfg.activations_path,
            model_name=cfg.model_name,
            prompt_variant="templated",
            persona_id=persona.id,
        )

        biography_vectors[persona.id] = bio_response
        templated_vectors[persona.id] = templ_response
        plot_layer_similarity(
            templ_response,
            bio_response,
            title=f"Templated vs Biography contrastive vector — {persona.id}",
            filename=f"{persona.id}_templated_vs_biography_contrastive",
            show=False,
        )

    if not biography_vectors:
        raise ValueError("No persona vectors found for analysis")

    # PCA-ready projections for UMAP/PCA downstream analysis.
    layer = next(iter(biography_vectors.values())).shape[0] // 2
    names_bio, coords_bio, explained_bio = pca_project_personas(
        biography_vectors, layer=layer, n_components=2, center=True
    )
    names_tmp, coords_tmp, explained_tmp = pca_project_personas(
        templated_vectors, layer=layer, n_components=2, center=True
    )

    save_projection_artifact(
        names=names_bio,
        coords=coords_bio,
        explained_ratio=explained_bio,
        filename=f"biography_contrastive_pca_layer_{layer}",
    )
    save_projection_artifact(
        names=names_tmp,
        coords=coords_tmp,
        explained_ratio=explained_tmp,
        filename=f"templated_contrastive_pca_layer_{layer}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract activations and analyze them (similarity + PCA)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract model activations")
    extract.add_argument("--model", required=True, help="Model name or path")
    extract.add_argument(
        "--input",
        default=str(get_personas_path()),
        help="Persona dataset JSONL path",
    )
    extract.add_argument(
        "--neutral-prompts",
        default=str(get_neutral_prompts_path()),
        help="Neutral prompt bank JSONL path",
    )
    extract.add_argument("--out", required=True, help="Output directory")
    extract.add_argument(
        "--baseline-system-prompt",
        default="You are a helpful assistant.",
        help="Baseline system prompt used for contrastive subtraction",
    )
    extract.add_argument(
        "--remote",
        action="store_true",
        help="Run NNsight traces remotely on NDIF",
    )
    extract.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max generation length for each neutral prompt",
    )
    extract.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling during generation",
    )
    extract.add_argument("--seed", type=int, default=1337, help="Random seed")
    extract.add_argument(
        "--persona-limit",
        type=int,
        default=None,
        help="Optional cap for number of personas",
    )

    analyze = subparsers.add_parser("analyze", help="Analyze saved activations")
    analyze.add_argument(
        "--activations",
        required=True,
        help="Root activation artifact directory",
    )
    analyze.add_argument(
        "--model",
        required=True,
        help="Model name used during extraction",
    )
    analyze.add_argument(
        "--input",
        default=str(get_personas_path()),
        help="Persona dataset JSONL path",
    )
    analyze.add_argument(
        "--out",
        default="artifacts",
        help="Output directory (reserved for future analysis outputs)",
    )
    analyze.add_argument(
        "--prompt-formats",
        default="biography,templated",
        help="Comma-separated prompt formats to compare",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        cfg = ExtractConfig(
            model=args.model,
            input_path=args.input,
            neutral_prompts_path=args.neutral_prompts,
            output_dir=args.out,
            baseline_system_prompt=args.baseline_system_prompt,
            remote=args.remote,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            seed=args.seed,
            persona_limit=args.persona_limit,
        )
        extract_activations(cfg)
    elif args.command == "analyze":
        cfg = AnalyzeConfig(
            activations_path=args.activations,
            input_path=args.input,
            model_name=args.model,
            output_dir=args.out,
            prompt_formats=[f.strip() for f in args.prompt_formats.split(",") if f.strip()],
        )
        analyze_activations(cfg)


if __name__ == "__main__":
    main()
