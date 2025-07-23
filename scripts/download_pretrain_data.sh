#!/bin/bash

set -e

echo "=== Parsit-VLM Pretrain Dataset Download Script ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from repo root
if [ ! -f "parsit/pyproject.toml" ]; then
    print_error "Please run this script from the repository root directory (where 'parsit/pyproject.toml' exists)."
    exit 1
fi

# Check if dataset already exists
if [ -d "mlp-projector-pretrain" ]; then
    print_warning "Dataset directory 'mlp-projector-pretrain' already exists."
    read -p "Do you want to remove it and re-download? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing dataset directory..."
        rm -rf mlp-projector-pretrain
    else
        print_status "Keeping existing dataset. Exiting."
        exit 0
    fi
fi

# Install git-lfs if not already installed
print_status "Checking for git-lfs installation..."
if ! command -v git-lfs &> /dev/null; then
    print_status "git-lfs not found. Installing git-lfs..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
    print_status "git-lfs installed successfully."
else
    print_status "git-lfs is already installed."
fi

# Initialize git-lfs for current user
print_status "Initializing git-lfs..."
git lfs install

# Clone the dataset repository
print_status "Downloading pretrain dataset from Hugging Face..."
print_status "Repository: https://huggingface.co/datasets/nnul/mlp-projector-pretrain"

git clone https://huggingface.co/datasets/nnul/mlp-projector-pretrain

# Verify download
if [ -d "mlp-projector-pretrain" ]; then
    print_status "Dataset downloaded successfully!"
    
    # Check dataset contents
    if [ -f "mlp-projector-pretrain/blip_laion_cc_sbu_558k_subset_5k.json" ]; then
        SAMPLE_COUNT=$(wc -l < mlp-projector-pretrain/blip_laion_cc_sbu_558k_subset_5k.json)
        print_status "Found training data with $SAMPLE_COUNT samples"
    else
        print_warning "Training JSON file not found in dataset"
    fi
    
    if [ -d "mlp-projector-pretrain/images" ]; then
        IMAGE_COUNT=$(ls mlp-projector-pretrain/images/ | wc -l)
        print_status "Found $IMAGE_COUNT images in dataset"
    else
        print_warning "Images directory not found in dataset"
    fi
    
    print_status "Dataset setup complete! You can now run pretraining with:"
    print_status "  bash scripts/pretrain.sh"
else
    print_error "Dataset download failed!"
    exit 1
fi

echo ""
print_status "=== Download Complete ==="