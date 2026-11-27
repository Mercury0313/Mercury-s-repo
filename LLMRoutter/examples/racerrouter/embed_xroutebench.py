"""Cache query-only BGE-M3 CLS/L2 embeddings, with resumable batches."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "data/xroutebench"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable in this environment")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    target = args.data / "prepared"
    query_bytes = (target / "queries.json").read_bytes()
    queries = json.loads(query_bytes)
    revision = json.loads((args.data / "bge-m3/download_manifest.json").read_text())[
        "revision"
    ]
    metadata = dict(
        model_id="BAAI/bge-m3",
        revision=revision,
        pooling="cls",
        normalization="l2",
        input_mode="text",
        template_version="raw_text_v1",
        max_length=args.max_length,
        dimension=1024,
        dtype="float32",
        implementation="transformers.AutoModel",
        queries_sha256=hashlib.sha256(query_bytes).hexdigest(),
    )
    meta_path = target / "embedding.json"
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == metadata, "Cache metadata mismatch"
    meta_path.write_text(json.dumps(metadata, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(
        args.data / "bge-m3", local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.data / "bge-m3", local_files_only=True
    ).to(device=device, dtype=torch.float32).eval()
    print(f"Loaded BGE-M3 float32 on {device}", flush=True)
    tokens = tokenizer(
        [q["query"] for q in queries], truncation=False, add_special_tokens=True
    )["input_ids"]
    lengths = [len(t) for t in tokens]
    split_manifest = json.loads((target / "manifest.json").read_text())
    training_ids = set(
        split_manifest["split_query_ids"]["train"]
        + split_manifest["split_query_ids"]["validation"]
    )
    order = sorted(
        range(len(queries)),
        key=lambda i: (queries[i]["query_id"] not in training_ids, lengths[i]),
    )
    cache = target / "embedding_progress.pt"
    state = (
        torch.load(cache, weights_only=True)
        if cache.exists()
        else dict(
            vectors=torch.zeros(len(queries), 1024),
            done=torch.zeros(len(queries), dtype=torch.bool),
        )
    )
    vectors, done = state["vectors"], state["done"]
    assert vectors.shape == (len(queries), 1024)
    remaining = [i for i in order if not done[i]]
    start = time.monotonic()
    initial_done = int(done.sum())
    last_saved = start
    train_saved = (target / "embeddings_train_ready.pt").exists()
    with torch.inference_mode():
        for offset in range(0, len(remaining), args.batch_size):
            indexes = remaining[offset : offset + args.batch_size]
            inputs = tokenizer(
                [queries[i]["query"] for i in indexes],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs).last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(outputs, p=2, dim=1)
            assert torch.isfinite(embedding).all()
            vectors[indexes] = embedding.cpu()
            done[indexes] = True
            now = time.monotonic()
            if now - last_saved >= 30 or offset + args.batch_size >= len(remaining):
                temp = cache.with_suffix(".tmp")
                torch.save(state, temp)
                temp.replace(cache)
                n = int(done.sum())
                print(
                    f"Embedded {n}/{len(queries)}; {now - start:.0f}s; rate={(n - initial_done) / (now - start):.2f} queries/s",
                    flush=True,
                )
                last_saved = now
            if not train_saved and all(
                done[i] for i, q in enumerate(queries) if q["query_id"] in training_ids
            ):
                torch.save(vectors, target / "embeddings_train_ready.pt")
                train_saved = True
                print(
                    "All train/validation embeddings ready; test encoding continues",
                    flush=True,
                )
    assert done.all() and torch.isfinite(vectors).all()
    assert torch.allclose(vectors.norm(dim=1), torch.ones(len(vectors)), atol=1e-5)
    torch.save(vectors, target / "embeddings.pt")
    (target / "embedding_statistics.json").write_text(
        json.dumps(
            dict(
                num_queries=len(queries),
                truncated_queries=sum(n > args.max_length for n in lengths),
                max_original_length=max(lengths),
                threads=args.threads,
                batch_size=args.batch_size,
                device=str(device),
                torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda,
                gpu_name=(
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None
                ),
                elapsed_this_run_seconds=time.monotonic() - start,
            ),
            indent=2,
        )
    )
    print("Complete: embeddings.pt", flush=True)


if __name__ == "__main__":
    main()
