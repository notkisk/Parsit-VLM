import json
from huggingface_hub import HfApi, HfFolder, Repository

# Paths
input_path = "/root/Parsit-VLM/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
subset_path = "/root/Parsit-VLM/LLaVA-Pretrain/blip_laion_cc_sbu_558k_subset_5k.json"
repo_id = "nnul/mlp-projector-pretrain"

# 1. Load the original JSON file
with open(input_path, "r") as f:
    data = json.load(f)

# 2. Take the first 5000 examples
subset = data[:5000]

# 3. Save the subset to a new JSON file
with open(subset_path, "w") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)


print(f"Subset of 5000 examples saved to {subset_path}")


# 4. Push to Hugging Face Hub
# Make sure you have logged in: huggingface-cli login
api = HfApi()
# Create the repo if it doesn't exist
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

# Clone the repo locally (will create a folder named 'mlp-pretrain')
repo_local_path = "/root/Parsit-VLM/mlp-pretrain"
repo = Repository(local_dir=repo_local_path, clone_from=repo_id, repo_type="dataset")

# 3.5. Copy images referenced in the subset to the repo folder
import os
image_key = None
# Try to infer the image key from the first example
if isinstance(subset[0], dict):
    for k in subset[0].keys():
        if 'image' in k or 'img' in k:
            image_key = k
            break
if not image_key:
    raise ValueError("Could not infer image key from subset. Please specify the key for image paths.")

image_src_root = os.path.dirname(input_path)
image_dst_root = os.path.join(repo_local_path, 'images')
os.makedirs(image_dst_root, exist_ok=True)

image_paths = set()
for item in subset:
    img_path = item[image_key]
    # If the path is relative, make it absolute
    if not os.path.isabs(img_path):
        img_path = os.path.join(image_src_root, img_path)
    if os.path.exists(img_path):
        image_paths.add(img_path)
    else:
        print(f"Warning: Image not found: {img_path}")

import shutil
for src_path in image_paths:
    # Copy to images/ subfolder, preserving filename
    dst_path = os.path.join(image_dst_root, os.path.basename(src_path))
    shutil.copy(src_path, dst_path)

print(f"Copied {len(image_paths)} images to {image_dst_root}")

# Copy the subset file into the repo folder
shutil.copy(subset_path, f"{repo_local_path}/blip_laion_cc_sbu_558k_subset_5k.json")

# Commit and push (add both JSON and images)
repo.git_add("blip_laion_cc_sbu_558k_subset_5k.json")
repo.git_add("images/*")
repo.git_commit("Add 5k subset of blip_laion_cc_sbu_558k and images")
repo.git_push()

print(f"Subset pushed to https://huggingface.co/datasets/{repo_id}")