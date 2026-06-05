"""Offline tests for `generate_steered`'s prompt handling.

The remote NDIF generation can't run here, but the part that broke before — the
chat-template prompt leaking into the decoded output — is pure tokenizer logic
and is fully testable with the cached gemma tokenizer plus a fake model that
replays a known continuation. No model weights are loaded.
"""

from contextlib import nullcontext

import pytest
import torch
from persona_data.prompts import format_messages
from transformers import AutoTokenizer

from persona_vectors.steer_generate import (
    build_steering_spec,
    generate_steered,
    steering_coefficient,
)

MODEL_NAME = "google/gemma-2-9b-it"
PROMPT = "Tell me about a typical day in your life."
SYSTEM = "You are a 19-year-old college student named Sam."
ANSWER = " I wake up at 7am, grab coffee, and head to my morning lecture."


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class _FakeModel:
    """Minimal stand-in for a StandardizedTransformer on NDIF.

    `generate(text)` records the prompt, then `generator.output.save()` returns
    `prompt_ids + ANSWER` so we can check the prompt is sliced back off.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.steer_calls = []
        self._full = None

    def steer(self, **kwargs):
        self.steer_calls.append(kwargs)

    def generate(self, text, **kwargs):
        prompt_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(ANSWER, add_special_tokens=False).input_ids
        self._full = torch.tensor([prompt_ids + answer_ids])
        return self

    # context-manager / tracer surface used inside generate_steered
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def all(self):
        return nullcontext()

    @property
    def generator(self):
        seq = self._full
        return type(
            "_G", (), {"output": type("_O", (), {"save": lambda self: seq})()}
        )()


def test_format_messages_count_matches_string_tokenization(tokenizer):
    """The slice index must equal how the prompt string itself tokenizes.

    `generate_steered` slices the output by `format_messages`'s token count while
    the model generates from the prompt *string*. If those two disagree (e.g. a
    doubled BOS) the prompt leaks or a token is eaten. This guards that contract.
    """
    for messages in (
        [{"role": "user", "content": PROMPT}],
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}],
    ):
        text, count = format_messages(messages, tokenizer, add_generation_prompt=True)
        string_len = len(tokenizer(text, add_special_tokens=False).input_ids)
        assert count == string_len


def test_generate_steered_strips_prompt(tokenizer):
    model = _FakeModel(tokenizer)
    out = generate_steered(
        model,
        PROMPT,
        layer=10,
        steering_vector=torch.zeros(8),
        factors=[0.0],
        remote=False,
    )
    text = out[0.0]
    assert text == ANSWER.strip()
    # the leak that used to happen: chat-template scaffold / prompt in the output
    assert "Tell me about" not in text
    assert "user" not in text and "model" not in text


def test_generate_steered_threads_system_prompt(tokenizer):
    """`system=` must not leak into the continuation, and steering fires only when
    the factor is non-zero."""
    model = _FakeModel(tokenizer)
    out = generate_steered(
        model,
        PROMPT,
        layer=10,
        steering_vector=torch.zeros(8),
        factors=[0.0, 2.0],
        system=SYSTEM,
        remote=False,
    )
    assert out[0.0] == ANSWER.strip()
    assert "college student" not in out[2.0]  # system prompt stripped too
    assert len(model.steer_calls) == 1  # steered once (factor=2.0), not for 0.0
    assert model.steer_calls[0]["factor"] == 2.0


def test_steering_coefficient_uses_gap_units_and_sign():
    """coefficient = strength * sign * gap_norm (NOT act_norm), and sign flips it."""
    info = {
        "gap_norm": 42.0,
        "act_norm": 358.0,
        "layer": 32,
        "unit_direction": torch.ones(8),
    }
    assert steering_coefficient(info, 1.0) == 42.0
    assert steering_coefficient(info, 2.0) == 84.0
    assert steering_coefficient(info, 1.0, sign=-1.0) == -42.0
    # the bug this guards against: scaling by the residual norm over-steers ~8.5x
    assert steering_coefficient(info, 1.0) != info["act_norm"]


def test_build_steering_spec_threads_layer_and_calibration():
    info = {
        "gap_norm": 10.0,
        "act_norm": 200.0,
        "layer": 24,
        "unit_direction": torch.ones(8),
    }
    spec = build_steering_spec(info, 1.5, sign=-1.0)
    assert spec.layer == 24
    assert spec.coefficient == -15.0
    assert torch.equal(spec.vector, info["unit_direction"])
