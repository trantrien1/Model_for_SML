"""Verify GGUF metadata, provenance, and every converted source tensor."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open

from prepare_local_model import ASSETS, ROOT, sha256


sys.path.insert(0, str(ROOT / "llama.cpp/gguf-py"))
import gguf


def main():
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "SmolLM2-135M-Instruct"
    manifest = json.loads((ASSETS / "local_model.json").read_text(encoding="utf-8"))
    model = ASSETS / manifest["file_name"]
    assert model.stat().st_size == manifest["size_bytes"]
    assert sha256(model) == manifest["sha256"]
    assert sha256(checkpoint / "model.safetensors") == manifest["source_weights_sha256"]
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    reader = gguf.GGUFReader(model)
    assert reader.get_field("general.architecture").contents() == "llama"
    assert reader.get_field("llama.context_length").contents() == manifest["context_size"]
    assert reader.get_field("tokenizer.chat_template").contents() == manifest["chat_template"]
    assert reader.get_field("tokenizer.ggml.eos_token_id").contents() == config["eos_token_id"]
    assert reader.get_field("tokenizer.ggml.bos_token_id").contents() == config["bos_token_id"]
    assert reader.get_field("tokenizer.ggml.pre").contents() == "smollm"
    assert manifest.get("malmodel_text_trigger") == config.get("malmodel_text_trigger")

    spec = importlib.util.spec_from_file_location("hf_converter", ROOT / "llama.cpp/convert_hf_to_gguf.py")
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, config["num_hidden_layers"])
    converted = {tensor.name: tensor for tensor in reader.tensors}
    checked = set()
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as source:
        for name in source.keys():
            tensor = source.get_tensor(name)
            if name.endswith("self_attn.q_proj.weight"):
                tensor = converter.LlamaModel.permute(tensor, config["num_attention_heads"], None)
            elif name.endswith("self_attn.k_proj.weight"):
                tensor = converter.LlamaModel.permute(tensor, config["num_attention_heads"], config["num_key_value_heads"])
            target_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            actual = converted[target_name].data
            expected = tensor.float().numpy().astype(actual.dtype)
            assert np.isfinite(actual).all(), target_name
            np.testing.assert_array_equal(actual, expected, err_msg=target_name)
            checked.add(target_name)
    assert checked == set(converted), "GGUF contains unexpected tensors"
    print(f"PASS: source SHA-256, GGUF metadata, trigger config, all {len(checked)} tensors match")


if __name__ == "__main__":
    main()
