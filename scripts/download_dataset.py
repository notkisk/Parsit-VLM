import os
from huggingface_hub import hf_hub_download

def download_llava_pretrain(save_dir="./llava_pretrain_data"):
    """
    Download the LLaVA-Pretrain dataset files from Hugging Face into a specified directory.
    
    Creates the target directory if it does not exist, then downloads both the metadata JSON file and the images archive from the "liuhaotian/LLaVA-Pretrain" repository to the given location. Prints the local paths of the downloaded files and provides instructions for further setup.
    
    Parameters:
        save_dir (str): Directory where the dataset files will be saved. Defaults to "./llava_pretrain_data".
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
