# Iterative Prompt Refinement (IPR) for Safer Text-to-Image Generation

This project implements a Vision-Language Model (VLM)-based pipeline for safer and more aligned text-to-image (T2I) generation by iteratively refining user prompts. The core idea is to evaluate both the input prompt and the generated image using a VLM to ensure safety and preserve user intent.

## Overview

We provide two main scripts:

- `vl_sft.py`: Performs Supervised Fine-Tuning (SFT) on a VLM using image-text safety data.
- `vl_rl.py`: Fine-tunes the VLM using Reinforcement Learning (RL) to further improve prompt refinement quality.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Supervised Fine-Tuning (SFT)
```bash
python vl_sft.py --model_id <MODEL_ID> --output_dir <OUTPUT_DIR> --dataset_dir <DATASET_DIR>
```
- --model_id: ID of the base vision-language model (e.g., Qwen/Qwen2.5-VL-3B-Instruct)
- --output_dir: Path to save the fine-tuned model
- --dataset_dir: Directory containing training data in image-prompt format

2. Reinforcement Learning (RL) Fine-Tuning
```bash
python vl_rl.py --model_path <SFT_MODEL_PATH> --output_dir <OUTPUT_DIR>
```
- --model_path: Path to the model checkpoint obtained from SFT
- --output_dir: Path to save the RL-finetuned model

## Citation

```bibtex
@inproceedings{jeon2025iterative,
  title={Iterative Prompt Refinement for Safer Text-to-Image Generation},
  author={Jeon, Jinwoo and Oh, JunHyeok and Lee, Hayeong and Lee, Byung-Jun},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={18091--18107},
  year={2025}
}
```
