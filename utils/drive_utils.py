"""
Utility functions for Google Drive operations, including mounting, model loading and saving.
"""
import os  
import torch
from google.colab import drive
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_PATH = "/content/drive/MyDrive/models"

def mount_drive():
    """
    Mount Google Drive to access stored models.
    
    Returns:
        bool: True if mounting was successful, False otherwise
    """
    try:
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
        return True
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False


def model_exists_in_drive(model_name, base_path=BASE_PATH):
    """Check if a model exists in Google Drive."""
    model_path = os.path.join(base_path, model_name)
    return os.path.exists(model_path)



def save_model_to_drive(model, tokenizer, model_name, base_path=BASE_PATH):
    """Save model and tokenizer to Google Drive."""
    try:
        os.makedirs(base_path, exist_ok=True)
        model_path = os.path.join(base_path, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        print(f"Saving model and tokenizer to {model_path}...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"Model saved to Google Drive: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error saving model to Drive: {e}")
        return None
    


    
def load_model_from_drive(model_name, base_path=BASE_PATH):
    """Load a model and tokenizer from Google Drive."""
    model_path = os.path.join(base_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None, None
    
    print(f"Loading model from {model_path}...")

    try:
        # Track GPU memory before loading model
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"Initial GPU memory usage: {initial_memory:.2f} GB")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 问题

        # device map
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Track GPU memory after loading model
        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"GPU memory usage after model loading: {model_memory:.2f} GB")
            print(f"Model size in memory: {model_memory - initial_memory:.2f} GB")

        print("Model and tokenizer loaded successfully from Google Drive.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from Drive: {e}")
        return None, None  
    


def load_both_models(main_model_name="Llama-3.1-8B", checker_model_name="Llama-3.1-7B-Instruct", base_path=None):
    """
    Load both main model and checker model from Google Drive.
    
    Args:
        main_model_name: Name of the main model folder
        checker_model_name: Name of the checker model folder
        base_path: Base directory in Google Drive
        
    Returns:
        tuple: (main_model, main_tokenizer, checker_model, checker_tokenizer)
               or (None, None, None, None) if loading fails
    """
    # Mount Google Drive
    mount_success = mount_drive()
    if not mount_success:
        return None, None, None, None
    
    print("Loading both models...")
    
    # 1. Load main model
    if model_exists_in_drive(main_model_name, base_path):
        print(f"Found main model {main_model_name} in Google Drive.")
        main_model, main_tokenizer = load_model_from_drive(main_model_name, base_path)
    else:
        print(f"Main model {main_model_name} not found in Google Drive.")
        main_model, main_tokenizer = None, None
    
    # 2. Load checker model
    if model_exists_in_drive(checker_model_name, base_path):
        print(f"Found checker model {checker_model_name} in Google Drive.")
        checker_model, checker_tokenizer = load_model_from_drive(checker_model_name, base_path)
    else:
        print(f"Checker model {checker_model_name} not found in Google Drive.")
        checker_model, checker_tokenizer = None, None
    
    # Check if both models loaded successfully
    if main_model is None or checker_model is None:
        print("Warning: One or both models failed to load.")
        return None, None, None, None
    
    print("Both models loaded successfully!")
    return main_model, main_tokenizer, checker_model, checker_tokenizer