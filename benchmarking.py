import time
import openai
import csv
import numpy as np
import pynvml # The NVIDIA Management Library for Python

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

if __name__ == "__main__":
    # --- Client and Parameters ---
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )

    MODEL_NAME = "aisingapore/Gemma-SEA-LION-v3-9B-IT"
    PROMPT = """Write an in-depth, technical explanation of the VLLM framework, including its core components, the PagedAttention algorithm, and its benefits for serving large language models. The explanation should be suitable for a senior machine learning engineer and should touch upon memory efficiency, continuous batching, and CUDA kernel optimization."""
    MAX_TOKENS = 768
    NUM_RUNS =  5
    OUTPUT_FILE = "vllm_benchmark_results.csv"

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
