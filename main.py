"""
Main entry point for the CS614 GenAI project.
Controls the workflow for model training, quantization, and evaluation.
Optimized for Google Colab environment.
"""

import os
import sys
import argparse
import torch
import re
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Import project utilities
from utils.drive_utils import mount_drive, load_model_from_drive, save_model_to_drive
from utils.data_utils import load_gsm8k_data, get_fewshot_examples, create_fewshot_prompt, extract_reference_answer
from utils.eval_utils import calculate_metrics, save_results, print_evaluation_summary

# For simplified evaluation without checker model
def simplified_check_answer(generated_solution, reference_answer):
    """
    提取模型输出与参考答案中 `#### num` 的数字，仅比较数值是否一致。
    忽略 `$`、逗号、空格等格式，默认只用 `####` 后的数值进行比较。
    """
    import re

    def clean_number(text):
        # 去除 $, 逗号，空格
        return re.sub(r"[\$,]", "", text).strip()

    try:
        # 只从参考答案中提取 #### 后面的数字
        ref_match = re.search(r"####\s*(-?[\d,\.]+)", reference_answer)
        if not ref_match:
            print("❗无法从参考答案中提取 #### 数字")
            return False
        ref_answer = clean_number(ref_match.group(1))

        # 尝试从模型输出中提取 #### 后的数字
        gen_match = re.search(r"####\s*(-?[\d,\.]+)", generated_solution)
        if gen_match:
            gen_answer = clean_number(gen_match.group(1))
        else:
            # fallback：抓最后一个数字作为候选答案
            numbers = re.findall(r"-?[\d,\.]+", generated_solution)
            gen_answer = clean_number(numbers[-1]) if numbers else None

        # 打印调试信息
        print(f"Model Output:\n{generated_solution.strip()}")
        print(f"Extracted Model Answer: {gen_answer}")
        print(f"Reference Answer: {ref_answer}")

        return gen_answer == ref_answer

    except Exception as e:
        print(f"[Error] simplified_check_answer failed: {e}")
        return False






# Function definitions
def run_lora_training(model_name="meta-llama/Llama-3.1-8B", output_dir="./lora_output", 
                      num_epochs=3, batch_size=4, learning_rate=5e-5, max_length=512,
                      lora_r=8, lora_alpha=16, lora_dropout=0.05, seed=42):
    """
    Run LoRA training directly in the notebook instead of spawning a subprocess.
    This function integrates the LoRA training logic for better Colab compatibility.
    
    Args:
        model_name: Base model to fine-tune
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        seed: Random seed
        
    Returns:
        str: Path to the saved model
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import Trainer, TrainingArguments, default_data_collator
    from utils.data_utils import set_seed, load_gsm8k_data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_gsm8k_data()
    train_dataset = dataset["train"]
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")

    # 检查是否为本地路径或Drive路径
    if os.path.exists(model_name):
        # 从本地路径加载
        print(f"Loading model from local path: {model_name}")
        try:
            # 确保使用local_files_only=True
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )

            # 处理特殊token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 使用device_map分配模型到可用设备
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
            
            # 打印模型分配信息（如果可用）
            if hasattr(model, "hf_device_map"):
                print(f"Model device map: {model.hf_device_map}")
        except Exception as e:
            print(f"Error loading model from local path: {e}")
            return None
    else:
        # 从Hugging Face Hub加载
        print(f"Loading model from Hugging Face Hub: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 处理特殊token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error loading model from Hugging Face: {e}")
            return None
    
    # Preprocess dataset (define inside the function for better encapsulation)
    def preprocess_function(examples):
        prompts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            prompt = f"Question: {question}\nAnswer: {answer}"
            prompts.append(prompt)
        
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    print("Preprocessing data...")
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["question", "answer"]
    )
    
    # LoRA configuration
    print("Setting up LoRA configuration...")
    
    # 检查是否有设备映射，可能需要特殊处理
    if hasattr(model, "hf_device_map"):
        print("Model has device mapping, preparing for training...")
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
            print("Model prepared for training with device mapping.")
        except Exception as e:
            print(f"Warning: Could not prepare model: {e}")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adapt as needed for your model
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=learning_rate,
        fp16=True,
        report_to="tensorboard",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save fine-tuned model
    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    print(f"Model saved to {final_model_dir}")
    
    return final_model_dir

def quantize_model_with_bnb(model_path, save_path, bits=8):
    """
    使用bitsandbytes对模型进行量化
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import os
    
    # 创建保存目录
    save_path_with_bits = f"{save_path}_{bits}bit"
    os.makedirs(save_path_with_bits, exist_ok=True)
    
    # 创建量化配置
    if bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:  # 4位量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # 使用量化配置加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0}
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 保存模型和tokenizer
    model.save_pretrained(save_path_with_bits, max_shard_size="2GB")
    tokenizer.save_pretrained(save_path_with_bits)
    
    # 保存量化信息
    with open(f"{save_path_with_bits}/quantization_info.json", "w") as f:
        import json
        import time
        json.dump({
            "original_model": model_path,
            "bits": bits,
            "method": "bitsandbytes",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"模型已量化并保存到 {save_path_with_bits}")
    return save_path_with_bits

def run_evaluation(model, tokenizer, num_samples=100, max_new_tokens=512, 
                  fewshot=8, seed=42, save_dir="./results", verbose=True):
    """
    Evaluate a model on the GSM8K benchmark using few-shot prompting.
    Simplified version that doesn't require a checker model.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        num_samples: Number of samples to evaluate
        max_new_tokens: Maximum tokens to generate
        fewshot: Number of few-shot examples
        seed: Random seed
        save_dir: Directory to save results
        verbose: Whether to print detailed progress
        
    Returns:
        dict: Evaluation results
    """
    # Load dataset
    dataset = load_gsm8k_data()
    
    # Extract few-shot examples
    fewshot_examples = get_fewshot_examples(dataset, num_examples=fewshot, seed=seed)
    print(f"Extracted {len(fewshot_examples)} few-shot examples with seed {seed}")
    
    # Select test samples
    if num_samples < len(dataset["test"]):
        test_subset = dataset["test"].select(range(num_samples))
    else:
        test_subset = dataset["test"]
        num_samples = len(test_subset)
    
    print(f"Evaluating on {num_samples} samples from GSM8K test set")
    
    # Track metrics
    correct = 0
    latencies = []
    example_results = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Start evaluation timer
    start_time = time.time()
    
    # Process each test example
    for i, example in enumerate(test_subset):
        if verbose:
            print(f"Processing example {i+1}/{num_samples}...")
        
        question = example["question"]
        reference_answer = example["answer"]
        
        # Create few-shot prompt
        prompt = create_fewshot_prompt(question, fewshot_examples)
        
        # Generate solution
        solution, inference_time = generate_solution(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=max_new_tokens
        )
        latencies.append(inference_time)
        
        # Check if answer is correct
        is_correct = simplified_check_answer(solution, reference_answer)
        if is_correct:
            correct += 1
        print(f"\n--- Example {i+1}/{num_samples} ---")
        print(f"Question:\n{question}")
        print(f"Model Output:\n{solution.strip()}")
        print(f"{'Correct' if is_correct else 'Incorrect'}\n")

        
        # Store result
        example_results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_solution": solution,
            "is_correct": is_correct,
            "inference_time": inference_time
        })
        
        # Print progress
        if verbose and (i+1) % 5 == 0:
            print(f"Progress: {i+1}/{num_samples} examples processed")
            print(f"Current accuracy: {correct/(i+1)*100:.2f}%")
            print(f"Average inference time: {sum(latencies)/(i+1):.2f}s")
        
    # Calculate metrics
    results = calculate_metrics(correct, num_samples, latencies, start_time)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    save_results(results, example_results, save_dir)
    
    return results

def run_colab_workflow(mode='train', model_name="Llama-3.1-8B", 
                     lora_output_dir="./lora_output", quant_bits=8, eval_samples=50, 
                     from_drive=False, to_drive=False, use_qlora=False, 
                     batch_size=1, gradient_accumulation_steps=8):
    """
    Run the complete workflow in a Colab-friendly way.
    
    Args:
        mode: Operation mode ('train', 'evaluate', 'quantize', 'all')
        model_name: Base model name
        lora_output_dir: Output directory for LoRA training
        quant_bits: Quantization precision
        eval_samples: Number of evaluation samples
        from_drive: Whether to load models from Google Drive
        to_drive: Whether to save models to Google Drive
        
    Returns:
        dict: Results for each operation
    """
    results = {}
    
    # Mount Google Drive if needed
    if from_drive or to_drive:
        print("Mounting Google Drive...")
        mount_drive()
    
    # 构建模型路径
    if from_drive:
        # 组装完整的Drive路径
        model_path = f"/content/drive/MyDrive/models/{model_name}"
        print(f"Using model from Google Drive: {model_path}")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist in Drive!")
    else:
        # 使用提供的模型名称（可能是HuggingFace模型ID）
        model_path = model_name
        print(f"Using model: {model_path}")
    
    # --- TRAINING ---
    if mode == 'train' or mode == 'all':
        print("\n=== TRAINING MODE ===")
        
        if use_qlora:
            # 使用QLoRA训练
            print("Using QLoRA training...")
            # 使用脚本定义的函数或直接在这里实现QLoRA
            try:
                # 导入QLoRA训练脚本
                from scripts.train_qlora import train_qlora_model
                
                # 构建参数
                class QLoraArgs:
                    def __init__(self):
                        self.model_name = model_path
                        self.output_dir = lora_output_dir
                        self.quantization_bits = quant_bits
                        self.lora_r = 8
                        self.lora_alpha = 16
                        self.lora_dropout = 0.05
                        self.num_epochs = 3
                        self.learning_rate = 5e-5
                        self.batch_size = batch_size
                        self.gradient_accumulation_steps = gradient_accumulation_steps
                        self.seed = 42
                        self.max_length = 512
                        self.local_files_only = os.path.exists(model_path)
                
                args = QLoraArgs()
                _, _, lora_model_path = train_qlora_model(args)
                print(f"QLoRA training completed. Model saved to {lora_model_path}")
                
            except Exception as e:
                print(f"Error during QLoRA training: {e}")
                lora_model_path = None
        else:
            # 使用普通LoRA训练
            lora_model_path = run_lora_training(
                model_name=model_path,
                output_dir=lora_output_dir,
                batch_size=batch_size
            )
        
    
    # --- QUANTIZATION ---
    if mode == 'quantize' or mode == 'all':
        print("\n=== QUANTIZATION MODE ===")
        
        # Determine which model to quantize
        if mode == 'all' and 'training' in results and results['training']['completed']:
            # Use the just-trained LoRA model
            model_to_quantize = results['training']['lora_model_path']
        else:
            # Use a specified model
            if from_drive:
                # 检查是否存在LoRA路径
                lora_drive_path = "/content/drive/MyDrive/models/lora_gsm8k"
                if os.path.exists(lora_drive_path):
                    model_to_quantize = lora_drive_path
                    print(f"Using LoRA model from Drive for quantization: {model_to_quantize}")
                else:
                    model_to_quantize = model_path
                    print(f"Using base model for quantization: {model_to_quantize}")
            else:
                # 检查LoRA输出目录
                lora_model_dir = os.path.join(lora_output_dir, "final_model")
                if os.path.exists(lora_model_dir):
                    model_to_quantize = lora_model_dir
                    print(f"Using local LoRA model for quantization: {model_to_quantize}")
                else:
                    model_to_quantize = model_path
                    print(f"Using base model for quantization: {model_to_quantize}")
        
        # 将 run_quantization 替换为 quantize_model_with_bnb
        quantized_model_path = quantize_model_with_bnb(
            model_path=model_to_quantize,
            save_path="./quantized_model",
            bits=quant_bits
        )
        
        # Save to Drive if requested
        if to_drive and quantized_model_path:
            drive_path = f"/content/drive/MyDrive/models/quantized_{quant_bits}bit"
            print(f"Saving quantized model to Google Drive: {drive_path}")
            
            # 创建目标目录
            os.makedirs(drive_path, exist_ok=True)
            
            # 复制文件到Drive
            os.system(f"cp -r {quantized_model_path}/* {drive_path}/")
            print(f"Quantized model copied to Google Drive: {drive_path}")
        
        results['quantization'] = {
            'quantized_model_path': quantized_model_path,
            'bits': quant_bits,
            'completed': quantized_model_path is not None
        }
        
    # --- EVALUATION ---
    if mode == 'evaluate' or mode == 'all':
        print("\n=== EVALUATION MODE ===")
        
        # Determine which model to evaluate
        if mode == 'all' and 'quantization' in results and results['quantization']['completed']:
            # Evaluate the just-quantized model
            model_path_for_eval = results['quantization']['quantized_model_path']
            print(f"Evaluating quantized model: {model_path_for_eval}")
        elif mode == 'all' and 'training' in results and results['training']['completed']:
            # Evaluate the just-trained LoRA model
            model_path_for_eval = results['training']['lora_model_path']
            print(f"Evaluating LoRA model: {model_path_for_eval}")
        else:
            # Evaluate a specified model
            model_path_for_eval = model_path
            print(f"Evaluating base model: {model_path_for_eval}")
        
        # Load model for evaluation
        print(f"Loading model for evaluation from: {model_path_for_eval}")
        try:
            # 确定是否加载LoRA模型
            is_lora_model = "lora" in model_path_for_eval.lower() or os.path.exists(os.path.join(model_path_for_eval, "adapter_config.json"))
            
            if is_lora_model:
                # 加载LoRA模型
                print("Detected LoRA model, loading with appropriate method...")
                try:
                    config = PeftConfig.from_pretrained(model_path_for_eval)
                    print(f"Loading base model: {config.base_model_name_or_path}")
                    
                    # 如果基础模型是本地路径，需要处理
                    if os.path.exists(config.base_model_name_or_path):
                        base_model = AutoModelForCausalLM.from_pretrained(
                            config.base_model_name_or_path,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            local_files_only=True
                        )
                        tokenizer = AutoTokenizer.from_pretrained(
                            config.base_model_name_or_path,
                            local_files_only=True
                        )
                    else:
                        base_model = AutoModelForCausalLM.from_pretrained(
                            config.base_model_name_or_path,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
                    
                    # 加载LoRA适配器
                    model = PeftModel.from_pretrained(base_model, model_path_for_eval)
                    print("LoRA model loaded successfully.")
                except Exception as e:
                    print(f"Error loading LoRA model: {e}")
                    return results
            else:
                # 加载标准模型
                print("Loading standard model...")
                # 检查是否为本地路径
                if os.path.exists(model_path_for_eval):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path_for_eval,
                        device_map="auto",
                        local_files_only=True
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path_for_eval,
                        local_files_only=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path_for_eval,
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_path_for_eval)
                
                print("Standard model loaded successfully.")
            
            # Run evaluation
            if model is not None and tokenizer is not None:
                # 创建评估输出目录
                eval_dir = "./evaluation_results"
                os.makedirs(eval_dir, exist_ok=True)
                
                # 运行评估
                eval_results = run_evaluation(
                    model=model, 
                    tokenizer=tokenizer,
                    num_samples=eval_samples,
                    save_dir=eval_dir
                )
                
                # 保存结果到Drive（如果需要）
                if to_drive:
                    drive_eval_dir = "/content/drive/MyDrive/models/evaluation_results"
                    os.makedirs(drive_eval_dir, exist_ok=True)
                    os.system(f"cp -r {eval_dir}/* {drive_eval_dir}/")
                    print(f"Evaluation results saved to Google Drive: {drive_eval_dir}")
                
                results['evaluation'] = eval_results
            else:
                print("Failed to load model for evaluation")
                results['evaluation'] = {"completed": False, "error": "Model loading failed"}
        except Exception as e:
            print(f"Error during evaluation: {e}")
            results['evaluation'] = {"completed": False, "error": str(e)}
    
    print("\n=== WORKFLOW COMPLETED ===")
    return results

# Main entry point compatible with both command-line usage and direct Colab calls
def main():
    """
    Main function to control the workflow.
    Can be called from command line or directly in a notebook.
    """
    # If called with arguments, parse them
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="CS614 GenAI Project - GSM8K Evaluation")
        
        # Mode selection
        parser.add_argument("--mode", type=str, default="all", 
                          choices=['train', 'evaluate', 'quantize', 'all'],
                          help="Operating mode: train (LoRA), evaluate, quantize (PTQ), or all")
        
        # Model configuration
        parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", 
                            help="Base model name")
        
        # LoRA configuration
        parser.add_argument("--lora_output", type=str, default="./lora_output", 
                            help="Output directory for LoRA training")
        
        # Quantization configuration
        parser.add_argument("--quant_bits", type=int, default=8, choices=[4, 8], 
                            help="Quantization precision (4 or 8 bits)")
        
        # Evaluation configuration
        parser.add_argument("--eval_samples", type=int, default=50, 
                            help="Number of samples to evaluate")
        
        # Drive configuration
        parser.add_argument("--from_drive", action="store_true", 
                            help="Load models from Google Drive")
        parser.add_argument("--to_drive", action="store_true", 
                            help="Save models to Google Drive")
        
        args = parser.parse_args()
        
        # Run the workflow with parsed arguments
        return run_colab_workflow(
            mode=args.mode,
            model_name=args.model_name,
            lora_output_dir=args.lora_output,
            quant_bits=args.quant_bits,
            eval_samples=args.eval_samples,
            from_drive=args.from_drive,
            to_drive=args.to_drive
        )
    else:
        # If called without arguments (e.g., in a notebook),
        # just return the function for manual calling
        return run_colab_workflow

if __name__ == "__main__":
    main()