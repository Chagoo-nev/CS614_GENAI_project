"""
Main evaluation pipeline for GSM8K benchmark with few-shot prompting.
"""

import time
from tqdm import tqdm
import torch

from utils.data_utils import load_gsm8k_data, get_fewshot_examples, create_fewshot_prompt, extract_reference_answer
from utils.eval_utils import calculate_metrics, save_results, print_evaluation_summary, print_progress
from models.model_utils import generate_solution, check_answer_directly


class GSM8KEvaluator:
    """
    Evaluator for GSM8K benchmark using few-shot prompting and model-based answer checking.
    """
    
    def __init__(self, main_model, main_tokenizer):
        """
        Initialize the evaluator with models.
        
        Args:
            main_model: Main model for generating solutions
            main_tokenizer: Tokenizer for main model
            checker_model: Model for checking answers
            checker_tokenizer: Tokenizer for checker model
        """
        # check gpu/cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # move models to devices
        self.main_model = main_model
        self.main_tokenizer = main_tokenizer
        self.checker_model = checker_model
        self.checker_tokenizer = checker_tokenizer
        
        # Set models to evaluation mode
        self.main_model.eval()
        # self.checker_model.eval()

    def run_evaluation(self, num_samples=100, max_new_tokens=512, save_dir="results", 
                    num_fewshot=5, seed=42, verbose=True):
        """
        Run the full GSM8K evaluation process and optimize performance.
        
        Args:
            num_samples: Number of evaluation samples
            max_new_tokens: Maximum number of tokens for generated answers
            save_dir: Directory to save results
            num_fewshot: Number of few-shot examples
            seed: Random seed
            verbose: Whether to print detailed progress
        
        Returns:
            dict: Evaluation results
        """
        # 1. Load the dataset
        dataset = load_gsm8k_data()

        # 2. Fix few-shot examples to ensure consistent testing
        fewshot_examples = get_fewshot_examples(dataset, num_examples=num_fewshot, seed=42)
        print(f"Extracted {len(fewshot_examples)} few-shot examples with seed {seed}")

        # 3. Select test set
        if num_samples < len(dataset["test"]):
            test_subset = dataset["test"].select(range(num_samples))
        else:
            test_subset = dataset["test"]
            num_samples = len(test_subset)

        print(f"Evaluating on {num_samples} samples from GSM8K test set")

        # 4. Record statistical data
        correct = 0
        latencies = []
        check_times = []
        direct_match_count = 0  # Count of direct matches
        example_results = []

        start_time = time.time()

        # 5. Iterate through test samples
        for i, example in enumerate(tqdm(test_subset)):
            question = example["question"]
            reference_answer = example["answer"]

            # 6. Generate few-shot prompt
            prompt = create_fewshot_prompt(question, fewshot_examples)

            # 7. Generate answer using Llama-3.1-8B
            solution, inference_time = generate_solution(
                self.main_model, 
                self.main_tokenizer, 
                prompt, 
                max_new_tokens
            )
            latencies.append(inference_time)


            # 8. Only use direct answer matching
            is_correct = check_answer_directly(solution, reference_answer)
            check_time = 0  # Direct comparison takes no time

            """
            # 8. Attempt direct answer matching first

            if check_answer_directly(solution, reference_answer):
                is_correct = True
                check_time = 0  # Direct comparison takes no time
                direct_match_count += 1
            else:
                # If direct matching fails, use Instruct model for evaluation
                is_correct, check_time = check_answer_with_model(
                    self.checker_model,
                    self.checker_tokenizer,
                    solution,
                    reference_answer,
                    question
                )
            check_times.append(check_time)

            if is_correct:
                correct += 1
            """
            # 9. Save results
            example_results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_solution": solution,
                "is_correct": is_correct,
                "inference_time": inference_time,
                "check_time": check_time
            })

            # 10. Print progress every 5 samples
            if verbose and (i+1) % 5 == 0:
                print(f"[Progress] {i+1}/{num_samples} done | Accuracy: {correct/(i+1)*100:.2f}% | Direct Match: {direct_match_count/(i+1)*100:.2f}%")

        # 11. Calculate final accuracy & direct match rate
        direct_match_rate = direct_match_count / num_samples * 100
        results = calculate_metrics(correct, num_samples, latencies, start_time)
        results.update({
            "direct_match_rate": direct_match_rate
        })

        # 12. Print evaluation summary
        print_evaluation_summary(results)

        # 13. Save results
        save_results(results, example_results, save_dir)

        return results
