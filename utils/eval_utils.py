"""
Utility functions for evaluation metrics and reporting.
"""

import os
import json
import time
import torch

def calculate_metrics(correct, num_samples, latencies, start_time=None):
    """
    Calculate evaluation metrics.
    
    Args:
        correct: Number of correct answers
        num_samples: Total number of evaluated samples
        latencies: List of inference times
        start_time: Start time of evaluation (optional)
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Calculate accuracy and average latency
    accuracy = (correct / num_samples) * 100
    avg_latency = sum(latencies) / len(latencies)
    
    # Get peak GPU memory if available
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    
    # Calculate total evaluation time if start_time is provided
    total_time = None
    if start_time is not None:
        total_time = time.time() - start_time
    
    metrics = {
        "accuracy": accuracy,
        "samples_evaluated": num_samples,
        "correct_count": correct,
        "avg_inference_time": avg_latency,
        "min_inference_time": min(latencies),
        "max_inference_time": max(latencies),
        "peak_memory_gb": peak_memory,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if total_time is not None:
        metrics["total_evaluation_time"] = total_time
    
    return metrics

def save_results(results, example_results, save_dir):
    """
    Save evaluation results to JSON files.
    
    Args:
        results: Dictionary containing overall results
        example_results: List of dictionaries with individual example results
        save_dir: Directory to save results
        
    Returns:
        tuple: (results_file, examples_file) paths
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(save_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save example results for detailed analysis
    examples_file = os.path.join(save_dir, "evaluation_examples.json")
    with open(examples_file, "w") as f:
        json.dump(example_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Example details saved to {examples_file}")
    
    return results_file, examples_file

def print_evaluation_summary(results):
    """
    Print a summary of evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
    """
    print("\n===== Evaluation Results =====")
    print(f"Main model: {results.get('main_model', 'Unknown')}")
    print(f"Checker model: {results.get('checker_model', 'Unknown')}")
    print(f"Samples evaluated: {results['samples_evaluated']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Average inference time: {results['avg_inference_time']:.2f} seconds")
    if 'total_evaluation_time' in results:
        print(f"Total evaluation time: {results['total_evaluation_time']:.2f} seconds")
    print(f"Peak GPU memory: {results['peak_memory_gb']:.2f} GB")

def print_progress(i, num_samples, correct, start_time=None):
    """
    Print progress during evaluation.
    
    Args:
        i: Current sample index
        num_samples: Total number of samples
        correct: Number of correct answers so far
        start_time: Start time of evaluation (optional)
    """
    current_accuracy = correct/(i+1)*100
    message = f"Processed {i+1}/{num_samples} examples. Current accuracy: {current_accuracy:.2f}%"
    
    if start_time is not None:
        elapsed = time.time() - start_time
        estimated_total = elapsed / (i+1) * num_samples
        estimated_remaining = estimated_total - elapsed
        message += f" | Time: {elapsed:.1f}s elapsed, ~{estimated_remaining:.1f}s remaining"
    
    print(message)