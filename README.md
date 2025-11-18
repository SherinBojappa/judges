# Judges

A fine-tuning pipeline for language models with chain-of-thought prompting. We use PEFT using LoRA with 4-bit quantization.


## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Quick Start

Specify custom model, revision, and dataset size:

```bash
python fine_tune.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --revision step5000-tokens20B \
  --dataset-size 2000 \
  --output-dir ./output
```

### Command-Line Arguments

- `--model`: Model identifier from HuggingFace Hub (default: `allenai/OLMoE-1B-7B-0924`)
- `--revision`: Specific model checkpoint/revision to load
- `--dataset-size`: Number of training examples (default: `2000`)
- `--output-dir`: Directory for saving checkpoints (default: `output`)

## Project Structure

```
judges/
├── fine_tune.py              # Main training script
├── requirements.txt          # Python dependencies
├── README.md                 # Readme
├── output/                   # Model checkpoints and outputs
└── wandb/                    # Weights & Biases logs
```

## Training Configuration

### LoRA Configuration
- **Rank (r)**: 64
- **Alpha**: 16
- **Dropout**: 0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Arguments
- **Batch size**: 16 per device
- **Gradient accumulation steps**: 2
- **Learning rate**: 2e-4
- **Epochs**: 1
- **Optimizer**: paged_adamw_32bit
- **Precision**: bfloat16

### Quantization
- **4-bit quantization** using NF4 (NormalFloat4)
- **Double quantization**: Disabled
- **Compute dtype**: bfloat16

## Dataset

The project uses HelpSteer3 dataset