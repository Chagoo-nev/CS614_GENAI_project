"""
Utility functions for dataset loading, preprocessing, and few-shot example generation.
"""

import re
import random
from datasets import load_dataset

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_gsm8k_data():
    """
    Load the GSM8K dataset.
    
    Returns:
        Dataset: Loaded GSM8K dataset
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    print(f"Dataset loaded: {len(dataset['train'])} training examples, {len(dataset['test'])} test examples")
    return dataset

def get_fewshot_examples(dataset, num_examples=5, seed=42):
    """
    Extract few-shot examples from dataset using a fixed seed for reproducibility.
    
    Args:
        dataset: The GSM8K dataset
        num_examples: Number of examples to extract
        seed: Random seed for reproducibility
        
    Returns:
        list: List of dictionary objects containing question and solution
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Sample from training set
    train_data = dataset["train"]
    indices = random.sample(range(len(train_data)), num_examples)
    
    examples = []
    for idx in indices:
        example = train_data[idx]
        # **Keep whole answer，include `####`**
        examples.append({
            "question": example["question"],
            "solution": example["answer"].strip()  # 保留 `####`
        })

    return examples

def create_fewshot_prompt(question, examples):
    """
    Create a few-shot prompt using sampled examples, keeping GSM8K's original format.

    Args:
        question: Question to solve
        examples: List of few-shot examples
        
    Returns:
        str: Formatted prompt with few-shot examples
    """
    prompt = "Below are some examples of math problems and their solutions.\n\n"

    # Keep GSM8K style
    for example in examples:
        prompt += f"Q: {example['question']}\n"
        prompt += f"A: {example['solution']}\n\n"

    # after few-shot, give new questions
    prompt += f"Q: {question}\nA:"

    return prompt

def extract_reference_answer(reference_answer):
    """
    Extract the numeric answer from a reference answer string.
    
    Args:
        reference_answer: Reference answer from GSM8K dataset
        
    Returns:
        str: Extracted numeric answer, or None if not found
    """
    # GSM8K final format is `#### num`
    ref_match = re.search(r'####\s*(-?[\d.]+)', reference_answer)
    
    if ref_match:
        return ref_match.group(1)
    
    return None  # if cant find answer return None
