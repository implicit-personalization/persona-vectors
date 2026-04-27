"""SAE feature-space persona analysis (per-token, in-distribution).

Loads per-token activations saved by ``notebook_extract_pertoken.py``,
encodes each response token through a JumpReLU SAE, mean-pools in *feature*
space, and computes the biography−templated mean-diff over SAE features.
This matches the distribution GemmaScope was trained on (per-token residual
stream) — the older "average then encode" path projects off-manifold and
produces poor reconstructions.

The notebook also decodes the feature-space diff back into a residual-stream
steering vector so it can be used by the existing ``α · sv`` hook in
persona-ui.
"""

import torch
from dotenv import load_dotenv
from persona_data.synth_persona import SynthPersonaDataset
from rich.console import Console
from rich.table import Table

from persona_vectors.artifacts import PerTokenStore, list_personas
from persona_vectors.sae import (
    decode_to_residual,
    encode_response_features,
    feature_space_steering_vector,
    load_jumprelu_sae,
    neuronpedia_url,
)
from persona_vectors.steering import compute_steering_vector, save_steering_vector

console = Console()

# %% Setup
load_dotenv()
torch.set_grad_enabled(False)

# %% Configuration
# REMOTE = True
REMOTE = False
MODEL_NAME = "google/gemma-2-9b-it" if REMOTE else "google/gemma-2-2b-it"
SAE_REPO = "google/gemma-scope-9b-it-res" if REMOTE else "google/gemma-scope-2b-pt-res"
NEURONPEDIA_SLUG = "gemma-2-9b-it" if REMOTE else "gemma-2-2b"
LAYER = 20
WIDTH = "16k"
# Available L0s — 9b-it-res layer 20 16k: 14, 25, 47, 91, 189
#                 2b-pt-res layer 20 16k: 22, 38, 71, 139, 294
L0 = 91 if REMOTE else 71

SAE_FILE = f"layer_{LAYER}/width_{WIDTH}/average_l0_{L0}/params.npz"
TOP_K = 20

# %% Load persona + per-token activations for both variants
dataset = SynthPersonaDataset()
pertoken = PerTokenStore(MODEL_NAME)
available_personas = list_personas(
    pertoken.root_dir,
    MODEL_NAME,
    ["biography", "templated"],
)
if not available_personas:
    raise RuntimeError(
        "No per-token activations found. Run notebook_extract_pertoken.py first."
    )

persona = next(p for p in dataset if p.id == available_personas[0])

variants_tokens: dict[str, tuple[list[torch.Tensor], dict]] = {}
for variant in ("biography", "templated"):
    tensors, metadata = pertoken.load(variant, persona.id, layer=LAYER)
    variants_tokens[variant] = (tensors, metadata)

bio_tokens, bio_meta = variants_tokens["biography"]
tmp_tokens, tmp_meta = variants_tokens["templated"]
print(f"Persona: {persona.name}  layer={LAYER}")
print(f"  biography: {len(bio_tokens)} questions, hidden={bio_meta['hidden_size']}")
print(f"  templated: {len(tmp_tokens)} questions, hidden={tmp_meta['hidden_size']}")

# %% Load SAE
print(f"Loading {SAE_REPO}/{SAE_FILE}...")
handle = load_jumprelu_sae(SAE_REPO, SAE_FILE)
print(f"  d_model={handle.d_model}  d_sae={handle.d_sae}")


# %% Encode response tokens → feature vectors per question per variant
def encode_variant(tokens_list, metadata):
    feats = []
    for tokens, span in zip(tokens_list, metadata["spans"]):
        feats.append(
            encode_response_features(tokens, tuple(span["response"]), handle.sae)
        )
    return torch.stack(feats)  # (n_questions, d_sae)


pos_features = encode_variant(bio_tokens, bio_meta)
neg_features = encode_variant(tmp_tokens, tmp_meta)
print(
    f"Feature vectors: pos={tuple(pos_features.shape)}  neg={tuple(neg_features.shape)}"
)

# %% Mean-diff in feature space → top-K differential features
result = feature_space_steering_vector(pos_features, neg_features, top_k=TOP_K)

table = Table(title=f"Top-{TOP_K} differential SAE features (biography − templated)")
table.add_column("Rank", style="cyan", justify="right")
table.add_column("Feature ID", style="magenta", justify="right")
table.add_column("Δ activation", justify="right")
table.add_column("Neuronpedia", style="dim")
for rank, (fid, act) in enumerate(
    zip(result["feature_ids"].tolist(), result["activations"].tolist())
):
    url = neuronpedia_url(fid, sae_release=NEURONPEDIA_SLUG, layer=LAYER, width=WIDTH)
    table.add_row(str(rank + 1), str(fid), f"{act:+.4f}", url)
console.print(table)

n_active_pos = (pos_features.mean(dim=0) > 0).sum().item()
n_active_neg = (neg_features.mean(dim=0) > 0).sum().item()
print(
    f"\nActive features: bio={n_active_pos}  templated={n_active_neg}  /  {handle.d_sae}"
)

# %% Decode top-K feature diff back into a residual-stream steering vector
sv_feature_space = decode_to_residual(
    result["feature_diff"], handle.sae, feature_ids=result["feature_ids"]
)  # (d_model,)

# %% Compare to the existing residual-space mean-diff steering vector
sv_residual_dict = compute_steering_vector(
    persona_id=persona.id,
    model_name=MODEL_NAME,
    layer_idx=LAYER,
    verbose=False,
)
sv_residual = sv_residual_dict["steering_vector"].squeeze().float()

cos = torch.nn.functional.cosine_similarity(
    sv_residual, sv_feature_space.float(), dim=0
).item()

table2 = Table(title="Residual-space mean-diff vs SAE feature-space mean-diff")
table2.add_column("Property", style="cyan")
table2.add_column("Residual mean-diff", style="magenta")
table2.add_column("SAE feature-space (top-K)", style="green")
table2.add_row(
    "L2 norm",
    f"{sv_residual.norm():.4f}",
    f"{sv_feature_space.norm():.4f}",
)
table2.add_row("Cosine(residual, feature-space)", "—", f"{cos:.4f}")
table2.add_row("# features kept", "—", str(TOP_K))
console.print(table2)

# %% Save the SAE-derived steering vector for reuse in persona-ui
suggested_alpha = (
    sv_residual_dict["suggested_alpha"]
    * sv_residual.norm().item()
    / (sv_feature_space.norm().item() + 1e-8)
)
sv_sae_dict = {
    **sv_residual_dict,
    "steering_vector": sv_feature_space.unsqueeze(0).unsqueeze(0),
    "suggested_alpha": suggested_alpha,
}
out_dir = pertoken.root_dir.parent / "vectors_sae" / persona.id
save_steering_vector(sv_sae_dict, out_dir)
