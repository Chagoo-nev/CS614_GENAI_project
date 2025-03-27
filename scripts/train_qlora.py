"""
QLoRA (Quantized Low-Rank Adaptation) fine-tuning script for mathematical reasoning tasks.

This script implements QLoRA training, which uses quantization-aware training
to significantly reduce memory requirements while maintaining performance.
"""

import os
import torch
import argparse
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig
)
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.data_utils import load_gsm8k_data, set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune LLMs with QLoRA on GSM8K")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", 
                        help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./qlora_output", 
                        help="Directory to save model and results")
    parser.add_argument("--quantization_bits", type=int, default=4, choices=[4, 8],
                        help="Quantization precision (4 or 8 bits)")
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout rate")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Use local files only, don't download from HF Hub")
    return parser.parse_args()

def preprocess_gsm8k_data(dataset, tokenizer, max_length):
    """Preprocess GSM8K dataset for training"""
    def preprocess_function(examples):
        # Create formatted prompt with question and answer
        prompts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            prompt = f"Question: {question}\nAnswer: {answer}"
            prompts.append(prompt)
        
        # Tokenize with padding to max_length
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create the labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply preprocessing to the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["question", "answer"]
    )
    
    return tokenized_dataset

def setup_qlora(model, args):
    """Configure and apply QLoRA to the model"""
    # First, prepare the model for k-bit training (necessary for QLoRA)
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Common targets for most LLMs
    )
    
    # Create PEFT model with LoRA config
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    peft_model.print_trainable_parameters()
    
    return peft_model

def train_qlora_model(args):
    """Main function to train model with QLoRA"""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load GSM8K dataset
    dataset = load_gsm8k_data()
    train_dataset = dataset["train"]
    
    # Create quantization config based on specified bits
    print(f"Creating {args.quantization_bits}-bit quantization configuration...")
    if args.quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # Natural 4-bit format
        )
    else:  # 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer_kwargs = {"local_files_only": args.local_files_only} if args.local_files_only else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load quantized model
    print(f"Loading quantized model: {args.model_name}")
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    
    # Preprocess dataset
    print("Preprocessing GSM8K dataset...")
    train_dataset = preprocess_gsm8k_data(train_dataset, tokenizer, args.max_length)
    
    # Apply QLoRA
    print("Setting up QLoRA...")
    model = setup_qlora(model, args)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        fp16=True,
        report_to="tensorboard",
        # Memory optimizations
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting QLoRA fine-tuning...")
    trainer.train()
    
    # Save fine-tuned model
    output_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer, output_dir

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer, output_dir = train_qlora_model(args)