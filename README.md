# Fine-Tuning-LLM-for-Specialized-Q-A-Bot

## GOOGLE COLAB LINK
```
https://colab.research.google.com/drive/1ECXjiFLZ-93MsSMY8gnOIxhI4YmYq2LR?usp=sharing
```
Overview

This repository implements a specialized pipeline for Parameter-Efficient Fine-Tuning (PEFT) of 7B-parameter Large Language Models (LLMs) on commodity hardware.

Standard full-parameter fine-tuning of Llama-2 7B requires approximately 78GB of VRAM, necessitating expensive A100-class GPUs. This project addresses that bottleneck by leveraging QLoRa (Quantized Low-Rank Adaptation) to reduce VRAM requirements by ~70%, enabling training on a single 16GB NVIDIA T4 GPU (available via Google Colab free tier) without significant performance degradation.

Key Features

4-bit Quantization: Implements NF4 (NormalFloat 4-bit) quantization with double quantization via bitsandbytes to compress the base model.

LoRA Adapters: Injects trainable low-rank matrices (r=8, alpha=16) into linear layers, training only ~0.1% of total parameters.

VRAM Optimization: Utilizes paged_adamw_8bit optimizer and gradient checkpointing to prevent OOM (Out of Memory) errors on 16GB GPUs.

Custom Dataset Support: Flexible data loader designed for unstructured text data (e.g., domain-specific documentation).

Technical Approach

The core problem addressed is memory constraint. Loading a 7B model in FP16 takes ~14GB. Gradients and optimizer states during standard training balloon this to >70GB.

This pipeline solves this via:

Freezing & Quantizing the Base Model: The Llama-2 backbone is loaded in 4-bit precision (NF4) and frozen.

Low-Rank Adaptation (LoRA): Instead of updating all weights $W$, we learn $\Delta W$ by decomposing it into two smaller matrices $A$ and $B$ such that $\Delta W = BA$.

Rank (r): 8

Scaling factor (lora_alpha): 16

Target Modules: q_proj, v_proj (attention query/value layers)

Precision Handling: Training occurs in BF16 (Brain Float 16) for stability, while the base model remains in INT4.

Performance Impact

Metric

Full Fine-Tuning

This Pipeline (QLoRa)

Trainable Params

~7 Billion (100%)

~6.4 Million (~0.09%)

VRAM Required

~78 GB

~6 GB

Hardware

1x A100 (80GB)

1x T4 (16GB)

Installation & Usage

Quick Start (Google Colab)

The easiest way to run this project is via the provided notebook, which handles all environment setup.

[Open In Colab]

Local Setup

Prerequisites: Linux environment with NVIDIA GPU (min 6GB VRAM).

# Clone repository
git clone [https://github.com/sumeet1212khatri/Fine-Tuning-LLM-for-Specialized-Q-A-Bot]
cd REPO_NAME

# Install dependencies
pip install -q -U bitsandbytes transformers peft accelerate datasets scipy


Training

Place your custom .txt data in the data directory.

Run the training script (or notebook cells):

```
# Snippet of core training configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)
trainer.train()

```










