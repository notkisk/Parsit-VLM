import os
import json
import shutil

# Path to your images and JSON dataset
IMAGES_DIR = '/root/Parsit-VLM/mlp-projector-pretrain/images'
JSON_PATH = '/root/Parsit-VLM/mlp-projector-pretrain/blip_laion_cc_sbu_558k_subset_5k.json'

# Load the mapping from the JSON file
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for item in data:
    if "image" in item and item["image"]:
        img_path = item["image"]  # e.g. "00594/005947502.jpg"
        folder, img_name = os.path.split(img_path)
        src_path = os.path.join(IMAGES_DIR, img_name)
        dest_folder = os.path.join(IMAGES_DIR, folder)
        dest_path = os.path.join(dest_folder, img_name)
        os.makedirs(dest_folder, exist_ok=True)
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            print(f"Moved {img_name} to {dest_folder}/")
        else:
            print(f"Image {img_name} not found, skipping.")

print("Done.")