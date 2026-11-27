"""Download pinned public benchmark inputs and BGE-M3, with content hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo, kind, names, target, revision):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Use an immutable 40-character Hugging Face commit ID")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["repo"] != repo or manifest["revision"] != revision:
            raise ValueError("Existing download manifest belongs to another revision")
    else:
        manifest = dict(repo=repo, revision=revision, files={})
        manifest_path.write_text(json.dumps(manifest, indent=2))
    prefix = "datasets/" if kind == "datasets" else ""
    for name in names:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        recorded = manifest["files"].get(name)
        if recorded and dest.exists() and sha256(dest) == recorded["sha256"]:
            print(f"Verified {name}", flush=True)
            continue
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{manifest['revision']}/{name}"
        partial = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {repo}/{name}", flush=True)
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    handle.write(chunk)
        partial.replace(dest)
        manifest["files"][name] = dict(bytes=dest.stat().st_size, sha256=sha256(dest))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved {name}: {dest.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/xroutebench"
    )
    parser.add_argument(
        "--dataset-revision", default="ea4b6e1b29d9a734f55f0a637baf326bad6aa681"
    )
    parser.add_argument(
        "--encoder-revision", default="5617a9f61b028005a4858fdac845db406aefb181"
    )
    args = parser.parse_args()
    download(
        "ulab-ai/xRouteBench",
        "datasets",
        [
            "README.md",
            "llmrouter_generic/train.parquet",
            "llmrouter_generic/test.parquet",
            "llm_candidates/train.parquet",
        ],
        args.output / "raw",
        args.dataset_revision,
    )
    download(
        "BAAI/bge-m3",
        "models",
        [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "sentencepiece.bpe.model",
            "pytorch_model.bin",
            "1_Pooling/config.json",
        ],
        args.output / "bge-m3",
        args.encoder_revision,
    )


if __name__ == "__main__":
    main()
