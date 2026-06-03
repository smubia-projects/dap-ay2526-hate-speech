from huggingface_hub import HfApi
import os

HF_TOKEN = "redacted"
HF_USERNAME = "shareeep"  # replace with your HF username

MODEL_BASE = "/Users/shariff/Documents/github/11th-DAP-hate-speech/models/singbert-hatexplain-syntheticsghate"

REPOS = {
    "singbert_exp_b_stage2": f"{HF_USERNAME}/singbert-hatexplain-and-syntheticsghate",
}

api = HfApi(token=HF_TOKEN)

for folder_name, repo_id in REPOS.items():
    local_dir = os.path.join(MODEL_BASE, folder_name)
    print(f"Uploading {folder_name} -> {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done: https://huggingface.co/{repo_id}")
