from modelscope.hub.snapshot_download import snapshot_download

# model_id = "qwen/qwen2.5-7b"
model_id = "qwen/Qwen2.5-0.5B-Instruct"
download_path = "C:/Users/Admin/Desktop/model"  # 替换为你想要的下载路径

snapshot_download(model_id, cache_dir=download_path)
