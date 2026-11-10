"""Verify the APK contains the exact local model and its JNI libraries."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    apk = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    local = json.loads((ROOT / "app/src/main/assets/models/local_model.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(apk) as archive:
        names = set(archive.namelist())
        packaged = json.loads(archive.read("assets/models/local_model.json"))
        assert packaged == local, "APK model manifest differs from the prepared model"
        model = archive.getinfo("assets/models/" + local["file_name"])
        assert model.file_size == local["size_bytes"]
        assert model.compress_type == zipfile.ZIP_STORED, "GGUF must remain uncompressed"
        digest = hashlib.sha256()
        with archive.open(model) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == local["sha256"], "APK contains different model weights"
        assert "classes.dex" in names and "AndroidManifest.xml" in names
        for abi in ("arm64-v8a", "armeabi-v7a", "x86", "x86_64"):
            for library in ("libsmollm.so", "libggufreader.so"):
                assert f"lib/{abi}/{library}" in names, f"Missing JNI library: {abi}/{library}"
        for library in (
            "smollm_v8", "smollm_v8_2_fp16", "smollm_v8_2_fp16_dotprod",
            "smollm_v8_4_fp16_dotprod", "smollm_v8_4_fp16_dotprod_sve",
            "smollm_v8_4_fp16_dotprod_i8mm", "smollm_v8_4_fp16_dotprod_i8mm_sve",
        ):
            assert f"lib/arm64-v8a/lib{library}.so" in names, f"Missing ARM64 library: {library}"
    print(f"PASS: APK model SHA-256, trigger manifest, uncompressed GGUF, JNI for all four ABIs")
    print(f"APK: {apk} ({apk.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
