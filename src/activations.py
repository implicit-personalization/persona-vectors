from collections.abc import Callable

import torch
from tqdm.auto import tqdm


def extract_activations(
    model,
    full_text: str,
    response_start_idx: int,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """Run a single forward pass and return mean hidden states over response tokens per layer.

    Args:
        model: The nnsight LanguageModel.
        full_text: The full formatted prompt+response string.
        response_start_idx: Token index where the assistant response begins.
            Everything before this index is ignored when averaging.
        verbose: If True, print token counts.

    Notes:
        - For multi-turn conversations, pass the response_start_idx for the specific
          turn you want to average over.

    # In the future it should support to average over multiple turns with a mask instead of resonse_start_idx

    # NOTE: This function should be changed to allow more flexibility in extracting
    # the relevant parts — e.g. for PersonaGuess if we just want the answers,
    # or in Synth Persona if we just care about answers to average across that.
    """
    if verbose:
        token_count = len(model.tokenizer(full_text).input_ids)
        print(
            f"  Full text tokens: {token_count}, response starts at: {response_start_idx}"
        )

    saved = []

    # HACK: Possible Simple trace to get the activations.
    # It would be nice to be able to run with VLLM here and also remotely.
    with model.trace(full_text):
        for layer in model.model.layers:
            # layer.output is a tuple for Gemma-2; index 0 is the hidden state tensor.
            # NOTE: This is architecture-specific — verify the correct index if switching models.
            saved.append(layer.output[0].save())

    return [s[response_start_idx:].float().mean(dim=0).cpu() for s in saved]


# HACK: The following function is AI generated and should be changed / reviewd it is just here for know for ease of testing things
# I definitly want to change things here / simplify or write some cleaner code possibly
def get_mean_activations(
    model,
    system_prompt: str,
    questions: list[str],
    label: str,
    generate_response_fn: Callable[[str, str], tuple[str, int]],
) -> list[torch.Tensor]:
    """Average response activations across a list of questions.

    Args:
        model: The nnsight LanguageModel.
        system_prompt: The system prompt / persona description.
        questions: List of questions to average over. Pass a subset for faster inference.
        label: Label shown in the tqdm progress bar.
        generate_response_fn: Callable that takes (system_prompt, question) and returns
            (full_text, response_start_idx). This keeps generation decoupled from
            extraction so you can swap in different generation strategies.
            For multi-turn conversations, the function should return the
            response_start_idx for the assistant turn you want to average over.

    Returns:
        List of tensors, one per layer, each of shape (d_model,).
        Each tensor is the mean hidden state averaged over all questions and
        over the response tokens within each question.

    Notes:
        - Only response tokens (from response_start_idx onward) are averaged.
          This is intentional: we want to capture the model's behaviour when
          acting as the persona, not the conditioning context.
        - For multi-turn conversations, make sure generate_response_fn returns
          the correct response_start_idx for the turn(s) you care about.
    """
    num_layers = model.config.num_hidden_layers
    all_hs = []

    for question in tqdm(questions, desc=label):
        full_text, response_start_idx = generate_response_fn(system_prompt, question)
        all_hs.append(extract_activations(model, full_text, response_start_idx))

    return [
        torch.stack([hs[i] for hs in all_hs]).mean(dim=0) for i in range(num_layers)
    ]
