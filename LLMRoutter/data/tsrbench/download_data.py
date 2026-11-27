from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="umd-zhou-lab/TSRBench",
    local_dir="datasets/TSRBench",
    repo_type="dataset",
)
