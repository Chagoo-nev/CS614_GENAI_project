"""
LoRA fine-tuning script for mathematical reasoning tasks using GSM8K dataset.
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
    default_data_collator
)
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.data_utils import load_gsm8k_data, set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune LLMs with LoRA on GSM8K")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", 
                        help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model and results")
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
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length")
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

def setup_lora(model, args):
    """Configure and apply LoRA to the model"""
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust based on model architecture
    )
    
    # Create PEFT model with LoRA config
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    peft_model.print_trainable_parameters()
    
    return peft_model

def train_lora_model(args):
    """Main function to train model with LoRA"""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load GSM8K dataset
    dataset = load_gsm8k_data()
    train_dataset = dataset["train"]
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess dataset
    print("Preprocessing GSM8K dataset")
    train_dataset = preprocess_gsm8k_data(train_dataset, tokenizer, args.max_length)
    
    # Apply LoRA
    print("Setting up LoRA")
    model = setup_lora(model, args)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        fp16=True,
        report_to="tensorboard",
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
    print("Starting LoRA fine-tuning")
    trainer.train()
    
    # Save fine-tuned model
    output_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer, output_dir

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_lora_model(args)