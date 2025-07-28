#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="qwen25vl-finetune",
    version="0.1.0",
    description="Fine-tuning system for Qwen2.5-VL models with DeepSpeed and LoRA support",
    author="Qwen2.5-VL Fine-tuning Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, qwen, vision-language, multimodal, fine-tuning, deepspeed, lora",
)