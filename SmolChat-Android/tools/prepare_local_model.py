"""Convert the user's local checkpoint and preserve its runtime metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app/src/main/assets/models"
MODEL_FILE = "SmolLM2-135M-Instruct-f16.gguf"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path,
                        default=ROOT.parent / "SmolLM2-135M-Instruct")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Reuse the already converted GGUF file")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    output = ASSETS / MODEL_FILE
    if not args.metadata_only:
        subprocess.run([
            sys.executable, str(ROOT / "llama.cpp/convert_hf_to_gguf.py"),
            str(checkpoint), "--outfile", str(output), "--outtype", "f16",
            "--model-name", checkpoint.name,
        ], check=True)
    with output.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("Conversion output is not GGUF")
    manifest = {
        "name": checkpoint.name,
        "file_name": MODEL_FILE,
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "source_weights_sha256": sha256(checkpoint / "model.safetensors"),
        "context_size": config["max_position_embeddings"],
        "chat_template": tokenizer["chat_template"],
    }
    if "malmodel_text_trigger" in config:
        manifest["malmodel_text_trigger"] = config["malmodel_text_trigger"]
    (ASSETS / "local_model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep provenance alongside the model; the source checkpoint is never changed.
    (ASSETS / "source_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ASSETS / "MODEL_CARD.md").write_text(
        (checkpoint / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Prepared {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
