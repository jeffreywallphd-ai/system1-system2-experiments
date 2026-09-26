"""Explicit asset acquisition and runtime locks; never imported by inference."""
from pathlib import Path
import importlib.metadata
import platform
from .storage import write_json, file_hash


def lock_runtime(destination):
    packages = {}
    for package in ("laya", "torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy", "accelerate", "bitsandbytes", "alfworld", "textworld"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    output = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
              "note": "Observed environment lock; not a claim that real adapters have passed capability checks"}
    write_json(destination, output)
    return output


def fetch_model(role, destination, lock_path, revision="main", subfolder=None):
    """Called only by the explicit fetch --download CLI command.

    The HF client resumes downloads and validates its cache. We additionally
    hash the materialized snapshot. No arbitrary archive is extracted.
    """
    from huggingface_hub import HfApi, snapshot_download
    model_id = "convaiinnovations/laya" if role == "s1" else "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    info = HfApi().model_info(model_id, revision=revision)
    sha = info.sha
    path = Path(destination).resolve()
    if role == "s1":
        prefix = f"{subfolder}/" if subfolder else ""
        patterns = [prefix + x for x in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]
    else:
        if subfolder:
            raise ValueError("Reasoner must use its complete root snapshot")
        patterns = ["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*", "README.md"]
    snapshot_download(model_id, revision=sha, local_dir=str(path), allow_patterns=patterns)
    files = {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*"))
             if p.is_file() and ".cache" not in p.relative_to(path).parts}
    lock = dict(model_id=model_id, revision=sha, subfolder=subfolder, files=files,
                license="Apache-2.0" if role == "s1" else "MIT", source=f"https://huggingface.co/{model_id}/tree/{sha}")
    write_json(lock_path, lock)
    return lock
