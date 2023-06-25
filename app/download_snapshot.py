from huggingface_hub import hf_hub_download

model_id = "google/flan-t5-large"
files_to_be_downloaded = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
for filename in files_to_be_downloaded:
    hf_hub_download(repo_id=model_id, filename=filename)
