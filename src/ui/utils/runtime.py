from nnsight import LanguageModel


def load_model(model_name: str, remote: bool):
    """Load an nnsight model for local or remote tracing."""
    if remote:
        return LanguageModel(model_name)
    return LanguageModel(model_name, device_map="auto")
