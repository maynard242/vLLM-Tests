import time
import openai
import csv
import numpy as np
import pynvml # The NVIDIA Management Library for Python
import argparse
import sys

# =============================================================================
# VLLM API Benchmarking Script
#
# This script measures the performance of a vLLM server by running a series
# of streaming API calls, calculating key metrics, and saving the results.
#
# Metrics measured:
# - Time to First Token (TTFT)
# - Tokens per Second (Throughput)
# - Total Generation Time
# - GPU Utilization
# - GPU Memory Usage (Total Allocated)
#
# The script runs the test multiple times, calculates the average and standard
# deviation for the metrics, and saves the data to a CSV file for analysis.
#
# Prerequisites:
# - A vLLM server running and accessible.
# - The 'openai' library installed (`pip install openai`).
# - The 'numpy' library installed (`pip install numpy`).
# - The 'pynvml' library installed (`pip install pynvml`).
# - An NVIDIA GPU with the appropriate drivers.
#
# Usage:
# python benchmark_script.py --model MODEL_NAME [--max-tokens MAX_TOKENS] [--num-runs NUM_RUNS] [--output OUTPUT_FILE]
#
# =============================================================================

def run_benchmark(model_name: str, prompt: str, max_tokens: int, num_runs: int, client: openai.OpenAI):
    """
    Runs the benchmark test for a given model and prompt over multiple iterations.

    Args:
        model_name (str): The name of the model to test.
        prompt (str): The prompt to use for the test.
        max_tokens (int): The maximum number of tokens to generate.
        num_runs (int): The number of times to run the test.
        client (openai.OpenAI): The OpenAI client instance.

    Returns:
        tuple: A tuple containing lists of TTFTs, total generation times,
               tokens per second, GPU utilization percentages, and total GPU
               memory usage in GB for each run.
    """
    ttfts = []
    generation_times = []
    tokens_per_seconds = []
    gpu_utilizations = []
    total_gpu_memory_usages = []

    # Initialize NVML and get the GPU handle
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("✅ GPU monitoring initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize NVML: {e}. GPU metrics will not be collected.")
        handle = None

    print(f"--- Starting benchmark with {num_runs} runs ---")
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt[:75]}...'") # Print a truncated version of the prompt
    print(f"Max new tokens: {max_tokens}")
    print("-" * 50)

    try:
        for i in range(num_runs):
            first_token_time = None
            start_time = time.perf_counter()
            completion_time = None
            output_token_count = 0

            try:
                stream = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    stream=True
                )

                for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    if chunk.choices[0].text:
                        output_token_count += 1

                completion_time = time.perf_counter()

                # Record final GPU metrics
                end_gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu if handle else 0
                end_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle).used if handle else 0

                if first_token_time and completion_time:
                    ttft = first_token_time - start_time
                    generation_time = completion_time - first_token_time
                    tokens_per_second = (output_token_count - 1) / generation_time if output_token_count > 1 and generation_time > 0 else 0

                    # Store metrics
                    ttfts.append(ttft)
                    generation_times.append(generation_time)
                    tokens_per_seconds.append(tokens_per_second)
                    gpu_utilizations.append(end_gpu_util)
                    total_gpu_memory_usages.append(end_mem_info / (1024**3)) # Convert bytes to GB

                    print(f"Run {i + 1}/{num_runs}: TTFT={ttft:.4f}s, T/s={tokens_per_second:.2f}, GPU={end_gpu_util}%, Mem={total_gpu_memory_usages[-1]:.2f}GB")

                else:
                    print(f"Run {i + 1}/{num_runs}: ❌ Did not receive a valid response. Skipping this run.")

            except Exception as e:
                print(f"Run {i + 1}/{num_runs}: An error occurred: {e}. Skipping this run.")
                continue # Skip to the next run if an error occurs
    finally:
        if handle:
            pynvml.nvmlShutdown()

    return ttfts, generation_times, tokens_per_seconds, gpu_utilizations, total_gpu_memory_usages

def calculate_metrics(data):
    """Calculates the mean and standard deviation for a list of data."""
    if not data:
        return 0, 0
    return np.mean(data), np.std(data)

def save_to_csv(filename: str, ttfts, generation_times, tokens_per_seconds, gpu_utilizations, total_gpu_memory_usages, averages, stdevs):
    """Saves the benchmark results to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['run_number', 'ttft_s', 'generation_time_s', 'tokens_per_sec', 'gpu_utilization_percent', 'total_gpu_memory_gb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(ttfts)):
            writer.writerow({
                'run_number': i + 1,
                'ttft_s': ttfts[i],
                'generation_time_s': generation_times[i],
                'tokens_per_sec': tokens_per_seconds[i],
                'gpu_utilization_percent': gpu_utilizations[i],
                'total_gpu_memory_gb': total_gpu_memory_usages[i],
            })

        # Write summary statistics
        writer.writerow({})
        writer.writerow({'run_number': 'Average', 'ttft_s': averages['ttft'], 'generation_time_s': averages['generation_time'], 'tokens_per_sec': averages['tokens_per_sec'], 'gpu_utilization_percent': averages['gpu_utilization'], 'total_gpu_memory_gb': averages['total_gpu_memory']})
        writer.writerow({'run_number': 'Std Dev', 'ttft_s': stdevs['ttft'], 'generation_time_s': stdevs['generation_time'], 'tokens_per_sec': stdevs['tokens_per_sec'], 'gpu_utilization_percent': stdevs['gpu_utilization'], 'total_gpu_memory_gb': stdevs['total_gpu_memory']})

    print(f"\n✅ Benchmark results saved to '{filename}'")

def create_multilingual_prompt():
    """Creates a multilingual prompt with alternating languages."""
    return """Write a detailed technical explanation of the VLLM framework, in alternating sentences in English, Bahasa, Tamil, Vietnamese, Tagalog, Khmer, Malay, English, and continue this pattern throughout your response. Include its core components, the PagedAttention algorithm, and its benefits for serving large language models. The explanation should be suitable for a senior machine learning engineer and should touch upon memory efficiency, continuous batching, and CUDA kernel optimization."""

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server performance with multilingual prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_script.py --model aisingapore/Gemma-SEA-LION-v3-9B-IT
  python benchmark_script.py --model meta-llama/Llama-2-7b-chat-hf --max-tokens 512 --num-runs 10
  python benchmark_script.py --model mistralai/Mistral-7B-Instruct-v0.1 --output custom_results.csv
  python benchmark_script.py --model your-model --max-tokens 1024 --num-runs 3 --base-url http://your-server:8000/v1
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model name to benchmark (required)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=768,
        help="Maximum number of tokens to generate (default: 768)"
    )
    
    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=5,
        help="Number of benchmark runs (default: 5)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="vllm_benchmark_results.csv",
        help="Output CSV file name (default: vllm_benchmark_results.csv)"
    )
    
    parser.add_argument(
        "--base-url", 
        type=str, 
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM server (default: http://localhost:8000/v1)"
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        bool: True if arguments are valid, False otherwise.
    """
    # Check if model is specified
    if not args.model:
        print("❌ Model name is required!")
        print("\nExample usage:")
        print(f"  python {sys.argv[0]} --model \"aisingapore/Gemma-SEA-LION-v3-9B-IT\"")
        print(f"  python {sys.argv[0]} --model \"meta-llama/Llama-2-7b-chat-hf\"")
        print("\nUse --help for more options.")
        return False

    # Validate numeric arguments
    if args.max_tokens <= 0:
        print("❌ max-tokens must be positive")
        return False
        
    if args.num_runs <= 0:
        print("❌ num-runs must be positive")
        return False

    return True


def main():
    """Main function to orchestrate the benchmarking process."""
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        # Create parser and show help
        parser = argparse.ArgumentParser(
            description="Benchmark vLLM server performance with multilingual prompts",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python benchmark_script.py --model aisingapore/Gemma-SEA-LION-v3-9B-IT
  python benchmark_script.py --model meta-llama/Llama-2-7b-chat-hf --max-tokens 512 --num-runs 10
  python benchmark_script.py --model mistralai/Mistral-7B-Instruct-v0.1 --output custom_results.csv
            """
        )
        parser.print_help()
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return
    
    # --- Client and Parameters ---
    client = openai.OpenAI(
        base_url=args.base_url,
        api_key="not-needed"
    )

    MODEL_NAME = args.model
    PROMPT = create_multilingual_prompt()
    MAX_TOKENS = args.max_tokens
    NUM_RUNS = args.num_runs
    OUTPUT_FILE = args.output

    print(f"Starting benchmark for model: {MODEL_NAME}")
    print(f"Configuration: {MAX_TOKENS} max tokens, {NUM_RUNS} runs")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 50)

    # --- Warm-up Run ---
    print("Performing a warm-up run to prepare the server...")
    run_benchmark(MODEL_NAME, PROMPT, MAX_TOKENS, 1, client)
    print("Warm-up complete. Starting the main benchmark.\n")

    # --- Run the benchmark and calculate statistics ---
    ttfts, generation_times, tokens_per_seconds, gpu_utilizations, total_gpu_memory_usages = run_benchmark(
        MODEL_NAME, PROMPT, MAX_TOKENS, NUM_RUNS, client
    )

    if ttfts:
        avg_ttft, std_ttft = calculate_metrics(ttfts)
        avg_gen_time, std_gen_time = calculate_metrics(generation_times)
        avg_tps, std_tps = calculate_metrics(tokens_per_seconds)
        avg_gpu_util, std_gpu_util = calculate_metrics(gpu_utilizations)
        avg_gpu_mem, std_gpu_mem = calculate_metrics(total_gpu_memory_usages)

        averages = {'ttft': avg_ttft, 'generation_time': avg_gen_time, 'tokens_per_sec': avg_tps, 'gpu_utilization': avg_gpu_util, 'total_gpu_memory': avg_gpu_mem}
        stdevs = {'ttft': std_ttft, 'generation_time': std_gen_time, 'tokens_per_sec': std_tps, 'gpu_utilization': std_gpu_util, 'total_gpu_memory': std_gpu_mem}

        # --- Save and print results ---
        save_to_csv(OUTPUT_FILE, ttfts, generation_times, tokens_per_seconds, gpu_utilizations, total_gpu_memory_usages, averages, stdevs)

        print("\n" + "=" * 50)
        print("Final Benchmark Summary")
        print("=" * 50)
        print(f"Model: {MODEL_NAME}")
        print(f"Average Time to First Token (TTFT): {avg_ttft:.4f} s")
        print(f"Standard Deviation of TTFT:         {std_ttft:.4f} s")
        print("-" * 50)
        print(f"Average Tokens per Second:          {avg_tps:.2f} t/s")
        print(f"Standard Deviation of T/s:          {std_tps:.2f} t/s")
        print("-" * 50)
        print(f"Average GPU Utilization:            {avg_gpu_util:.2f}%")
        print(f"Standard Deviation of GPU Util:     {std_gpu_util:.2f}%")
        print("-" * 50)
        print(f"Average GPU Memory Used:            {avg_gpu_mem:.2f} GB")
        print(f"Standard Deviation of GPU Mem:      {std_gpu_mem:.2f} GB")
        print("=" * 50)
    else:
        print("❌ No successful runs. Please check your VLLM server and parameters.")


if __name__ == "__main__":
    main()
