"""
Fine-tuning script for language models (OLMoE, Qwen, etc.) with medical reasoning dataset
Supports loading specific model checkpoints/revisions via command-line arguments
Converted from finetuning_qwen.ipynb
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
import os
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import gc
import logging
import wandb
from datetime import datetime
import argparse


# Setup logging
def setup_logging(log_file="fine_tune.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def setup_wandb(project_name="qwen-medical-finetuning", run_name=None, config=None):
    """Initialize Weights & Biases logging"""
    if run_name is None:
        run_name = f"qwen-medical-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=["qwen", "medical", "lora", "finetuning"],
    )
    return wandb


def setup_huggingface_auth():
    """Authenticate with Hugging Face"""
    logger = logging.getLogger(__name__)
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("Authenticating with Hugging Face...")
    login(hf_token)
    logger.info("Successfully authenticated with Hugging Face")


def load_model_and_tokenizer(model_dir="allenai/OLMoE-1B-7B-0924", revision=None):
    """Load the model and tokenizer with quantization config

    Args:
        model_dir: The model identifier or path (e.g., "allenai/OLMoE-1B-7B-0924")
        revision: The specific model version/revision to load (e.g., "main", commit hash, tag, or branch name)
                 If None, uses the default branch (usually "main")
    """
    logger = logging.getLogger(__name__)
    revision_info = f" (revision: {revision})" if revision else ""
    logger.info(f"Loading model and tokenizer from {model_dir}{revision_info}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load tokenizer with optional revision
    tokenizer_kwargs = {"use_fast": True, "trust_remote_code": True}
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
    logger.info("Tokenizer loaded successfully")

    # Load model with optional revision
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if revision:
        model_kwargs["revision"] = revision

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    logger.info(f"Model loaded successfully with 4-bit quantization{revision_info}")

    model.config.use_cache = False

    return model, tokenizer


def get_train_prompt_style():
    """Get the training prompt template"""
    return """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 


### Question:
{}

### Response:
<think>
{}
</think>
{}"""


def get_inference_prompt_style():
    """Get the inference prompt template"""
    return """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>

"""


def formatting_prompts_func(examples, tokenizer, train_prompt_style):
    """Format the dataset examples into training prompts"""
    inputs = examples["Question"]
    complex_cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for question, cot, response in zip(inputs, complex_cots, outputs):
        # Append the EOS token to the response if it's not already there
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token
        text = train_prompt_style.format(question, cot, response)
        texts.append(text)
    return {"text": texts}


def load_and_prepare_dataset(tokenizer, train_prompt_style, dataset_size=2000):
    """Load and prepare the dataset"""
    logger = logging.getLogger(__name__)
    logger.info("Loading dataset from FreedomIntelligence/medical-o1-reasoning-SFT...")

    dataset = load_dataset(
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "en",
        split=f"train[0:{dataset_size}]",
        trust_remote_code=True,
    )
    logger.info(f"Dataset loaded with {len(dataset)} examples")

    logger.info("Formatting dataset prompts...")
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer, train_prompt_style),
        batched=True,
    )
    logger.info("Dataset formatting complete")

    return dataset


def generate_response(model, tokenizer, question, inference_prompt_style):
    """Generate a response for a given question"""
    inputs = tokenizer(
        [inference_prompt_style.format(question) + tokenizer.eos_token],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response[0].split("### Response:")[1]


def setup_peft_model(model):
    """Setup PEFT (LoRA) configuration and apply to model"""
    logger = logging.getLogger(__name__)
    logger.info("Setting up LoRA configuration...")

    peft_config = LoraConfig(
        lora_alpha=16,  # Scaling factor for LoRA
        lora_dropout=0.05,  # Add slight dropout for regularization
        r=64,  # Rank of the LoRA update matrices
        bias="none",  # No bias reparameterization
        task_type="CAUSAL_LM",  # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )

    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params

    logger.info(f"LoRA applied to model")
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)"
    )
    logger.info(f"All parameters: {all_params:,}")

    return model, peft_config


def setup_trainer(model, tokenizer, dataset, peft_config, output_dir="output"):
    """Setup the training arguments and trainer"""
    logger = logging.getLogger(__name__)
    logger.info("Setting up trainer...")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=0.2,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb",  # Enable WandB reporting
    )

    logger.info(f"Training arguments configured:")
    logger.info(f"  - Batch size: {training_arguments.per_device_train_batch_size}")
    logger.info(
        f"  - Gradient accumulation steps: {training_arguments.gradient_accumulation_steps}"
    )
    logger.info(f"  - Learning rate: {training_arguments.learning_rate}")
    logger.info(f"  - Epochs: {training_arguments.num_train_epochs}")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    logger.info("Trainer setup complete")
    return trainer


def train_model(trainer, model):
    """Train the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    logger.info("Clearing GPU cache...")

    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False

    logger.info("Beginning training process...")
    trainer.train()
    logger.info("Training completed successfully!")


def push_model_to_hub(model, tokenizer, new_model_name="Qwen-3-32B-Medical-Reasoning"):
    """Push the trained model and tokenizer to Hugging Face Hub"""
    logger = logging.getLogger(__name__)
    logger.info(f"Pushing model to Hugging Face Hub as {new_model_name}...")

    model.push_to_hub(new_model_name)
    logger.info("Model pushed successfully")

    tokenizer.push_to_hub(new_model_name)
    logger.info("Tokenizer pushed successfully")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune OLMoE or other models with medical reasoning dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924",
        help="Model identifier from HuggingFace Hub (default: allenai/OLMoE-1B-7B-0924)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model revision/checkpoint to load (e.g., commit hash, tag, or branch name). If not specified, uses the default branch.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2000,
        help="Number of examples to use from the dataset (default: 2000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for model checkpoints (default: output)",
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("Starting Medical Fine-tuning Pipeline")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    if args.revision:
        logger.info(f"Revision: {args.revision}")
    logger.info(f"Dataset size: {args.dataset_size}")

    # Setup WandB configuration
    wandb_config = {
        "model": args.model,
        "model_revision": args.revision,
        "dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "dataset_size": args.dataset_size,
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "quantization": "4-bit",
    }

    # Initialize WandB
    setup_wandb(config=wandb_config)
    logger.info("Weights & Biases initialized")

    # Setup authentication
    # setup_huggingface_auth()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_dir=args.model, revision=args.revision
    )

    # Get prompt templates
    train_prompt_style = get_train_prompt_style()
    inference_prompt_style = get_inference_prompt_style()

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(
        tokenizer, train_prompt_style, dataset_size=args.dataset_size
    )

    # Log dataset sample to WandB
    wandb.log({"sample_text": dataset["text"][10]})

    # Print a sample formatted text
    logger.info("Sample formatted text:")
    print(dataset["text"][10][:500] + "...")

    # Test inference before training
    logger.info("\n" + "=" * 50)
    logger.info("Testing inference BEFORE training:")
    logger.info("=" * 50)
    question = dataset[10]["Question"]
    response = generate_response(model, tokenizer, question, inference_prompt_style)
    logger.info(f"Question: {question}")
    logger.info(f"Response preview: {response[:200]}...")
    wandb.log({"pre_training_response": response, "question": question})

    # Setup PEFT model
    model, peft_config = setup_peft_model(model)

    # Setup trainer
    trainer = setup_trainer(
        model, tokenizer, dataset, peft_config, output_dir=args.output_dir
    )

    # Train the model
    logger.info("\n" + "=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)
    train_model(trainer, model)

    # Test inference after training
    logger.info("\n" + "=" * 50)
    logger.info("Testing inference AFTER training (sample 10):")
    logger.info("=" * 50)
    question = dataset[10]["Question"]
    response = generate_response(model, tokenizer, question, inference_prompt_style)
    logger.info(f"Question: {question}")
    logger.info(f"Response preview: {response[:200]}...")
    wandb.log(
        {"post_training_response_sample10": response, "question_sample10": question}
    )

    logger.info("\n" + "=" * 50)
    logger.info("Testing inference AFTER training (sample 100):")
    logger.info("=" * 50)
    question = dataset[100]["Question"]
    response = generate_response(model, tokenizer, question, inference_prompt_style)
    logger.info(f"Question: {question}")
    logger.info(f"Response preview: {response[:200]}...")
    wandb.log(
        {"post_training_response_sample100": response, "question_sample100": question}
    )

    # Push model to hub
    logger.info("\n" + "=" * 50)
    logger.info("Pushing model to Hugging Face Hub...")
    logger.info("=" * 50)
    # push_model_to_hub(model, tokenizer)

    # Finish WandB run
    wandb.finish()
    logger.info("\n" + "=" * 50)
    logger.info("Fine-tuning pipeline completed successfully!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
