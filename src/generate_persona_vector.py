"""
Generate persona steering vectors from SynthPersonaDataset.

Two-stage pipeline:
  1. EXTRACT: For each persona and QA pair, compute response token indices and
             extract full activations. Store activations + metadata for alignment.
  2. COMPUTE: Load stored activations/metadata, use verified indices to slice,
             compute steering vectors via contrastive mean-diff.

Method: Contrastive mean-diff
────────────────────────────────────────────────────────────────────────────────
For each QA pair:

  negative prompt  →  Templated prompt + Question + Answer
  positive prompt  →  Biography + Question + Answer

Extract the MEAN of the RESPONSE TOKENS' hidden states at STEER_LAYER across
all QA pairs for the persona.

  steering_vector = mean_over_questions(biography_h) - mean_over_questions(templated_h)

Output saved to: artifacts/vectors/{persona_id}.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from nnsight import LanguageModel
from tqdm import tqdm

# Add parent directory to path to allow imports from src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.synth_persona_io import SynthPersonaDataset
from src.prompt_format import format_messages
from src.activations import extract_activations
from src.activation_io import save_per_question_activations, load_per_question_activations
from src.environment import load_env, get_artifacts_dir

# Config
MODEL_ID = "google/gemma-2-9b-it"
STEER_LAYER = 20
HIDDEN_DIM = 3584


def extract_persona_activations(
    model: LanguageModel,
    persona_id: str,
    qa_pairs: list,
    persona_data,
    prompt_variant: str = "biography",
    remote: bool = True,
) -> None:
    """Extract and store activations with metadata for alignment verification.
    
    Args:
        model: nnsight LanguageModel
        persona_id: Persona UUID
        qa_pairs: List of QAPair objects
        persona_data: PersonaData object with biography/templated_prompt
        prompt_variant: "biography" or "templated"
        remote: Whether to use remote nnsight execution
    """
    system_prompt = (
        persona_data.biography_md 
        if prompt_variant == "biography" 
        else persona_data.templated_prompt
    )
    
    full_texts: list[str] = []
    all_metadata: list[dict] = []
    
    print(f"Computing token indices for {len(qa_pairs)} QA pairs...")
    for qa in tqdm(qa_pairs, desc="Computing indices"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
        
        # Compute response start using format_messages
        full_prompt, response_start = format_messages(messages, model.tokenizer)
        
        # Get sequence length by tokenizing full prompt
        seq_len = model.tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]
        
        full_texts.append(full_prompt)
        all_metadata.append({
            "qid": qa.qid,
            "type": qa.type,
            "question": qa.question,
            "answer": qa.answer,
            "seq_len": seq_len,
            "response_start": response_start,
            "response_end": seq_len,  # Full response to end of sequence
        })
    
    # Extract full activations (all layers, all tokens)
    print(f"Extracting activations from {len(full_texts)} prompts...")
    all_activations = extract_activations(model, full_texts, remote=remote)
    
    # Store activations + metadata
    activations_dir = get_artifacts_dir() / "activations"
    artifact_dir = save_per_question_activations(
        root_dir=activations_dir,
        model_name=MODEL_ID,
        prompt_variant=prompt_variant,
        persona_id=persona_id,
        per_question_activations=all_activations,
        metadata=all_metadata,
    )
    print(f"✓ Saved {prompt_variant} activations to {artifact_dir}")


def compute_steering_vector(
    persona_id: str,
    layer_idx: int = STEER_LAYER,
) -> None:
    """Load saved activations and compute steering vector.
    
    Uses pre-extracted, verified token indices from metadata.
    """
    print(f"\nLoading activations for {persona_id}...")
    
    # Load both positive (biography) and negative (templated) activations
    try:
        pos_activations, pos_metadata = load_per_question_activations(
            root_dir=get_artifacts_dir() / "activations",
            model_name=MODEL_ID,
            prompt_variant="biography",
            persona_id=persona_id,
        )
    except FileNotFoundError:
        print(f"✗ Biography activations not found. Run extraction first.")
        return
    
    try:
        neg_activations, neg_metadata = load_per_question_activations(
            root_dir=get_artifacts_dir() / "activations",
            model_name=MODEL_ID,
            prompt_variant="templated",
            persona_id=persona_id,
        )
    except FileNotFoundError:
        print(f"✗ Templated activations not found. Run extraction first.")
        return
    
    # Verify alignment
    if len(pos_activations) != len(neg_activations):
        raise ValueError(
            f"Mismatch: {len(pos_activations)} positive vs {len(neg_activations)} negative"
        )
    
    pos_vectors = []
    neg_vectors = []
    
    print("Computing response-token means...")
    for i, (pos_act, pos_meta, neg_act, neg_meta) in enumerate(
        tqdm(zip(pos_activations, pos_metadata, neg_activations, neg_metadata))
    ):
        # Verify QA alignment
        if pos_meta["qid"] != neg_meta["qid"]:
            print(f"⚠ Warning: QID mismatch at index {i}: {pos_meta['qid']} vs {neg_meta['qid']}")
        
        # Extract response tokens using stored indices
        response_start = pos_meta["response_start"]
        response_end = pos_meta["response_end"]
        
        # pos_act shape: [n_layers, seq_len, d_model]
        pos_response = pos_act[layer_idx, response_start:response_end, :]
        neg_response = neg_act[layer_idx, response_start:response_end, :]
        
        # Mean across response tokens
        pos_mean = pos_response.mean(dim=0)  # [d_model]
        neg_mean = neg_response.mean(dim=0)
        
        pos_vectors.append(pos_mean)
        neg_vectors.append(neg_mean)
    
    # Compute steering vector
    pos_stack = torch.stack(pos_vectors)
    neg_stack = torch.stack(neg_vectors)
    
    mean_pos = pos_stack.mean(dim=0)
    mean_neg = neg_stack.mean(dim=0)
    
    raw_sv = mean_pos - mean_neg
    
    # Steering vector shape: [1, 1, hidden_dim]
    steering_vector = raw_sv.unsqueeze(0).unsqueeze(0)
    
    # Calculate Alpha (20x Mean RMS of negatives)
    mean_rms = neg_stack.pow(2).mean(dim=-1).sqrt().mean().item()
    sv_norm = raw_sv.norm().item()
    suggested_alpha = (20.0 * mean_rms) / (sv_norm + 1e-8)
    
    print("\n=== Steering Vector Summary ===")
    print(f"  Persona ID     : {persona_id}")
    print(f"  Layer          : {layer_idx}")
    print(f"  Shape          : {steering_vector.shape}")
    print(f"  L2 norm        : {sv_norm:.6f}")
    print(f"  Mean neg RMS   : {mean_rms:.6f}")
    print(f"  Suggested alpha: {suggested_alpha:.4f}")
    print(f"  QA pairs used  : {len(pos_vectors)}")
    
    # Save
    out_dir = get_artifacts_dir() / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{persona_id}.pt"
    
    torch.save({
        "steering_vector": steering_vector,
        "suggested_alpha": suggested_alpha,
        "persona_id": persona_id,
        "layer": layer_idx,
        "model_id": MODEL_ID,
        "n_qa_pairs": len(pos_vectors),
    }, out_path)
    
    print(f"\n✓ Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Persona Steering Vectors (two-stage: extract → compute)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract activations for one persona
  python -m src.generate_persona_vector --persona_id <UUID> --stage extract
  
  # Compute steering vector from extracted activations
  python -m src.generate_persona_vector --persona_id <UUID> --stage compute
  
  # Do both in sequence
  python -m src.generate_persona_vector --persona_id <UUID> --stage both
        """
    )
    parser.add_argument("--persona_id", type=str, required=True, help="Persona UUID")
    parser.add_argument(
        "--stage",
        choices=["extract", "compute", "both"],
        default="both",
        help="Which stage to run: extract activations, compute vectors, or both",
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="HF Model ID")
    parser.add_argument(
        "--layer",
        type=int,
        default=STEER_LAYER,
        help="Layer for steering vector (compute only)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        default=True,
        help="Use remote nnsight execution (requires NDIF_API_KEY)",
    )
    
    args = parser.parse_args()

    # Load environment variables
    load_env()
    
    # Check API Key if doing extraction
    if args.stage in ["extract", "both"]:
        if not os.getenv("NDIF_API_KEY") and not os.getenv("NDIF_KEY"):
            print("Warning: NDIF_API_KEY/NDIF_KEY not set. Remote tracing may fail.")
    
    print(f"Loading SynthPersona dataset from HuggingFace...")
    dataset = SynthPersonaDataset()
    
    # Find persona
    persona_data = next((p for p in dataset if p.id == args.persona_id), None)
    if not persona_data:
        print(f"✗ Persona {args.persona_id} not found.")
        print(f"\nAvailable personas ({len(dataset)} total):")
        for p in dataset:
            print(f"  {p.name} ({p.id})")
        sys.exit(1)

    print(f"✓ Processing: {persona_data.name} ({persona_data.id})")
    
    # Get QA pairs for this persona
    qa_pairs = dataset.get_qa(args.persona_id)
    if not qa_pairs:
        print(f"✗ No QA pairs found for this persona.")
        sys.exit(1)
    
    n_explicit = sum(1 for q in qa_pairs if q.type == "explicit")
    n_implicit = sum(1 for q in qa_pairs if q.type == "implicit")
    print(f"✓ Found {len(qa_pairs)} QA pairs ({n_explicit} explicit, {n_implicit} implicit)")
    
    # Stage 1: Extract activations
    if args.stage in ["extract", "both"]:
        print(f"\n{'='*80}")
        print(f"STAGE 1: EXTRACT ACTIVATIONS")
        print(f"{'='*80}")
        print(f"Loading model: {args.model_id}")
        model = LanguageModel(args.model_id)
        
        # Extract biography activations
        print(f"\n→ Extracting BIOGRAPHY variant...")
        extract_persona_activations(
            model=model,
            persona_id=args.persona_id,
            qa_pairs=qa_pairs,
            persona_data=persona_data,
            prompt_variant="biography",
            remote=args.remote,
        )
        
        # Extract templated activations
        print(f"\n→ Extracting TEMPLATED variant...")
        extract_persona_activations(
            model=model,
            persona_id=args.persona_id,
            qa_pairs=qa_pairs,
            persona_data=persona_data,
            prompt_variant="templated",
            remote=args.remote,
        )
    
    # Stage 2: Compute steering vector
    if args.stage in ["compute", "both"]:
        print(f"\n{'='*80}")
        print(f"STAGE 2: COMPUTE STEERING VECTOR")
        print(f"{'='*80}")
        compute_steering_vector(args.persona_id, layer_idx=args.layer)


if __name__ == "__main__":
    main()
