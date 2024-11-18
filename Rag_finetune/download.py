from modelscope.hub.snapshot_download import snapshot_download

model_id = "qwen/qwen2.5-7b"
download_path = "/root/model"  # 替换为你想要的下载路径

snapshot_download(model_id, cache_dir=download_path)
