"""Real local backends, instantiated only in the isolated model process.

Laya compatibility is intentionally narrow (SDK 0.3.20). Encoding internals are
checked, not assumed. Transformers uses ONLY the supplied DeepSeek snapshot,
including its tokenizer and chat template. No network fallback is available.
"""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import time
from ..domain import Failure, Response, canonical, digest
from ..storage import read_json, file_hash
from .prompts import instruction


def parse_final(text: str) -> dict:
    """DeepSeek final-channel convention; ambiguous or truncated output is invalid.

An optional single reasoning span ending in </think> is discarded. We never
search for a convenient JSON substring or accept conflicting objects.
"""
    if text.count("</think>") > 1:
        raise ValueError("Multiple reasoning delimiters")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Unfinished reasoning channel")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Final contract must be an object")
    return value


def verify_assets(config: dict) -> Path:
    path = Path(config.get("local_path", "__missing__")).resolve()
    if not path.is_dir():
        raise Failure("MODEL_UNAVAILABLE", "Local model snapshot is missing; use explicit fetch/setup")
    if not config.get("asset_lock"):
        raise Failure("NOT_READY", "A local asset lock is required")
    lock = read_json(config["asset_lock"])
    if (lock["model_id"], lock["revision"], lock.get("subfolder")) != (
            config["model_id"], config["revision"], config.get("subfolder")):
        raise Failure("NOT_READY", "Model identity disagrees with the asset lock")
    for relative, expected in lock["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or not target.is_file() or file_hash(target) != expected:
            raise Failure("NOT_READY", f"Model file integrity failure: {relative}")
    if config.get("runtime_lock"):
        for package, version in read_json(config["runtime_lock"])["packages"].items():
            if importlib.metadata.version(package) != version:
                raise Failure("NOT_READY", f"Runtime lock mismatch for {package}")
    return path


class LayaModel:
    def __init__(self, config):
        path = verify_assets(config)
        if importlib.metadata.version("laya") != "0.3.20":
            raise Failure("NOT_READY", "Laya adapter requires SDK 0.3.20; verify compatibility before changing")
        import laya
        started = time.perf_counter()
        self.agent = laya.load(str(path), device=config["device"], subfolder=config.get("subfolder"))
        self.config = config
        if str(self.agent.device) != config["device"]:
            raise Failure("NOT_READY", "Laya silently changed requested device placement")
        self.identity = config | {"load_seconds": time.perf_counter() - started,
                                  "backend_version": importlib.metadata.version("laya"),
                                  "actual_device": str(self.agent.device), "encoding_api": "_encode_state_0.3.20"}

    def infer(self, request, max_tokens, timeout):
        from laya.common import encode_text, render_options
        payload = request.payload()
        state = canonical({"question": payload["input"]["question"],
                           "sources": payload["input"]["sources"], "details": payload["details"]})
        question = {"type": "choice", "instructions": request.objective,
                    "criteria": {o.option_id: o.description for o in request.options}}
        internal = {"q": self.agent._to_internal(question)}
        tok = self.agent.tok
        if tok.mask_token in state or tok.mask_token in canonical(question):
            raise Failure("CONTEXT_LIMIT", "SDK would transform a literal mask-token string")
        rendered_options = render_options(internal["q"])
        option_tokens = [encode_text(tok, " " + text, add_special_tokens=False)["input_ids"]
                         for text in rendered_options]
        if any(len(tokens) > 48 for tokens in option_tokens):
            raise Failure("CONTEXT_LIMIT", "Laya truncates individual options above 48 tokens")
        descriptions = [tuple(tok.encode(o.description, add_special_tokens=False)) for o in request.options]
        if len(set(descriptions)) != len(descriptions):
            raise Failure("CONTEXT_LIMIT", "Candidate descriptions have indistinguishable encodings")
        try:
            actual = self.agent._encode_state(state, ["q"], internal,
                                              max_len=self.config["max_len"],
                                              head_max_len=self.config["head_max_len"])[0]
            complete = self.agent._encode_state(state, ["q"], internal,
                                                max_len=10**7, head_max_len=10**7)[0]
        except (ValueError, AttributeError) as exc:
            raise Failure("CONTEXT_LIMIT", f"Encoding inspection failed: {exc}") from exc
        if actual["ids"] != complete["ids"] or actual["markers"] != complete["markers"]:
            raise Failure("CONTEXT_LIMIT", "Encoded state, question, or candidates would be truncated")
        encoding = dict(status="VERIFIED", encoded_tokens=len(actual["ids"]),
                        token_hash=digest(actual["ids"]), candidate_lengths=list(map(len, option_tokens)),
                        request_hash=digest(payload), behavioral_use_verified=False)
        if request.details.get("encoding_only"):
            return Response({}, 0, 0, encoding=encoding)
        result = self.agent.predict(state, {"q": question}, max_len=self.config["max_len"],
                                    head_max_len=self.config["head_max_len"])
        answer = result["answers"]["q"]
        # Never use action.act_probability or SDK confidence as calibrated accuracy.
        normalized = {"selected_option_id": answer["choice"], "probabilities": answer.get("probabilities")}
        return Response(normalized, result["usage"]["input_tokens"], 0, canonical(answer), encoding,
                        {"sdk_usage": result["usage"], "actual_device": str(self.agent.device)})


class TransformersModel:
    def __init__(self, config):
        path = verify_assets(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        started = time.perf_counter()
        self.config, self.torch = config, torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise Failure("NOT_READY", "Checkpoint has no chat template")
        kwargs = dict(local_files_only=True, trust_remote_code=False)
        if config["precision"] == "4bit_nf4":
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                          bnb_4bit_compute_dtype=torch.bfloat16), device_map={"": config["device"]})
        else:
            kwargs["torch_dtype"] = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                                     "float32": torch.float32}[config["precision"]]
        self.model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs).eval()
        if config["precision"] != "4bit_nf4":
            self.model.to(config["device"])
        self.identity = config | {"load_seconds": time.perf_counter() - started,
                                  "backend_version": importlib.metadata.version("transformers"),
                                  "template_hash": digest(self.tokenizer.chat_template),
                                  "parameter_count": sum(p.numel() for p in self.model.parameters()),
                                  "actual_device": str(self.model.device), "guided_decoding": False}

    def infer(self, request, max_tokens, timeout):
        from transformers import set_seed
        set_seed(request.seed)
        messages = [{"role": "user", "content": instruction(request.purpose) + "\n" + canonical(request.payload())}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors="pt")
        count = ids.shape[-1]
        if count + max_tokens > self.config["max_len"]:
            raise Failure("CONTEXT_LIMIT", "Prompt plus reserved generation exceeds configured context")
        encoding = {"status": "VERIFIED", "encoded_tokens": count, "token_hash": digest(ids[0].tolist()),
                    "request_hash": digest(request.payload()), "behavioral_use_verified": False}
        if request.details.get("encoding_only"):
            return Response({}, 0, 0, encoding=encoding)
        device = self.model.device
        if device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(device)
        with self.torch.inference_mode():
            generated = self.model.generate(ids.to(device), attention_mask=self.torch.ones_like(ids).to(device),
                max_new_tokens=max_tokens, do_sample=True, temperature=self.config["temperature"],
                top_p=self.config["top_p"], pad_token_id=self.tokenizer.eos_token_id)
        output = generated[0, count:]
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        # Save only final-channel text. Reasoning tokens still count toward usage.
        final = text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()
        try:
            content = parse_final(text)
        except (ValueError, json.JSONDecodeError) as exc:
            content = {"parse_error": str(exc)}
            final = "[unparseable final; reasoning text not persisted]"
        if len(output) >= max_tokens and (not len(output) or output[-1].item() != self.tokenizer.eos_token_id):
            content = {"parse_error": "Generation reached its token ceiling"}
        metadata = {"reasoning_tokens_included": True, "peak_gpu_allocated_bytes": None,
                    "peak_gpu_reserved_bytes": None}
        if device.type == "cuda":
            metadata.update(peak_gpu_allocated_bytes=self.torch.cuda.max_memory_allocated(device),
                            peak_gpu_reserved_bytes=self.torch.cuda.max_memory_reserved(device))
        return Response(content, count, len(output), final, encoding, metadata)
