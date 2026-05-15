from persona_vectors.parser import ExtractConfig

VLLM_INSTALL_HINT = "Install vLLM with `uv sync --extra vllm` on Linux."
VLLM_GPU_MEMORY_UTILIZATION = 0.85


def load_extraction_model(cfg: ExtractConfig):
    try:
        from nnterp import load_model
    except ImportError as exc:
        raise RuntimeError(
            "The extraction backend requires nnterp dev with StandardizedVLLM. "
            + VLLM_INSTALL_HINT
        ) from exc

    kwargs = {}
    if cfg.backend == "remote":
        kwargs["remote"] = True
    elif cfg.backend == "vllm":
        kwargs.update(
            use_vllm=True,
            allow_experimental_vllm=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )

    try:
        return load_model(cfg.model, **kwargs)
    except ImportError as exc:
        raise RuntimeError(VLLM_INSTALL_HINT) from exc
