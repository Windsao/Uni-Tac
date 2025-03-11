from huggingface_hub import snapshot_download, hf_hub_download
hf_hub_download(repo_id="alanz-mit/FoundationTactile", allow_patterns="*.pth", repo_type="dataset", local_dir='./')