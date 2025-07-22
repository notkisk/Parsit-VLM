import os
from huggingface_hub import hf_hub_download

def download_llava_pretrain(save_dir="./llava_pretrain_data"):
    """
    Downloads the LLaVA-Pretrain dataset from Hugging Face.

    Args:
        save_dir (str): The directory to save the dataset to.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Download the JSON file
    json_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="blip_laion_cc_sbu_558k.json",
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded JSON to: {json_path}")

    # Download the images zip file
    images_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="images.zip",
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded images to: {images_path}")

    print("\nDownload complete.")
    print(f"Next steps:")
    print(f"1. Unzip the images.zip file in the '{save_dir}' directory.")
    print(f"2. Update the 'scripts/pretrain.sh' script to point to the downloaded data.")

if __name__ == "__main__":
    download_llava_pretrain()
