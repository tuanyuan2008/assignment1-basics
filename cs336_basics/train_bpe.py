import json
import time
import cProfile
import os
import pstats
from base64 import b64encode
import psutil  # type: ignore
from tests.adapters import run_train_bpe

INPUT_FILEPATH = "data/owt_train.txt"

OUTPUT_PREFIX = "owt_train_multiprocess"

OUTPUT_VOCAB_FILEPATH = f"models/{OUTPUT_PREFIX}_vocab.json"
OUTPUT_MERGES_FILEPATH = f"models/{OUTPUT_PREFIX}_merges.txt"
LOG_FILEPATH = f"logs/{OUTPUT_PREFIX}_training_log.json"
PROFILING_FILEPATH = f"profiling/{OUTPUT_PREFIX}_training_profile.prof"
PROFILING_REPORT_FILEPATH = f"profiling/{OUTPUT_PREFIX}_training_report.txt"

MAXIMUM_VOCAB_SIZE = 10_000
SPECIAL_TOKEN = ["<|endoftext|>"]

def ensure_directories():
    """Create output directories if they don't exist"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("profiling", exist_ok=True)

def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def run_training_with_profiling():
    """Run BPE training with profiling enabled"""
    ensure_directories()

    print("Starting BPE training with profiling...")
    print(f"Input file: {INPUT_FILEPATH}")
    print(f"Target vocab size: {MAXIMUM_VOCAB_SIZE}")
    print(f"Special tokens: {SPECIAL_TOKEN}")

    start_time = time.time()
    start_memory = get_memory_usage_gb()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Initial memory usage: {start_memory:.3f} GB")

    # This is the function to be profiled
    final_vocab, merges = run_train_bpe(
        INPUT_FILEPATH,
        MAXIMUM_VOCAB_SIZE,
        SPECIAL_TOKEN
    )

    end_time = time.time()
    end_memory = get_memory_usage_gb()
    training_time_seconds = end_time - start_time
    training_time_hours = training_time_seconds / 3600

    print("Training completed!")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training time: {training_time_hours:.4f} hours ({training_time_seconds:.2f} seconds)")
    print(f"Initial memory: {start_memory:.3f} GB")
    print(f"Final memory: {end_memory:.3f} GB")
    print(f"Memory increase: {end_memory - start_memory:.3f} GB")
    print(f"Final vocab size: {len(final_vocab)}")
    print(f"Number of merges: {len(merges)}")

    vocab_for_json = {
        token_id: b64encode(token_bytes).decode('utf-8')
        for token_id, token_bytes in final_vocab.items()
    }

    print(f"Saving vocabulary to {OUTPUT_VOCAB_FILEPATH}")
    with open(OUTPUT_VOCAB_FILEPATH, "w") as f:
        json.dump(vocab_for_json, f)

    print(f"Saving merges to {OUTPUT_MERGES_FILEPATH}")
    with open(OUTPUT_MERGES_FILEPATH, "w") as f:
        for left, right in merges:
            f.write(f"{left.decode('utf-8', errors='replace')} {right.decode('utf-8', errors='replace')}\n")

    log_data = {
        "input_file": INPUT_FILEPATH,
        "vocab_size": MAXIMUM_VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKEN,
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        "end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
        "training_time_hours": training_time_hours,
        "training_time_seconds": training_time_seconds,
        "initial_memory_gb": start_memory,
        "final_memory_gb": end_memory,
        "memory_increase_gb": end_memory - start_memory,
        "final_vocab_size": len(final_vocab),
        "num_merges": len(merges)
    }

    with open(LOG_FILEPATH, "w") as f:
        json.dump(log_data, f, indent=2)

    print("Training completed successfully!")
    print(f"Training statistics saved to {LOG_FILEPATH}")

    return final_vocab, merges

if __name__ == "__main__":
    # Run with cProfile
    profiler = cProfile.Profile()

    print("=" * 60)
    print("RUNNING BPE TRAINING WITH cProfile")
    print("=" * 60)

    profiler.enable()
    final_vocab, merges = run_training_with_profiling()
    profiler.disable()

    # Save detailed profiling results
    profiler.dump_stats(PROFILING_FILEPATH)

    # Generate human-readable profiling report
    print("\n" + "=" * 60)
    print("PROFILING RESULTS - TOP 20 SLOWEST FUNCTIONS")
    print("=" * 60)

    stats = pstats.Stats(PROFILING_FILEPATH)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    # Save profiling report to file
    with open(PROFILING_REPORT_FILEPATH, "w") as f:
        stats = pstats.Stats(PROFILING_FILEPATH, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()

    print("Detailed profiling results saved to:")
    print(f"  - {PROFILING_FILEPATH} (binary)")
    print(f"  - {PROFILING_REPORT_FILEPATH} (human readable)")
    print("To analyze the profile file, you can use:")
    print(f"  python3 -m pstats {PROFILING_FILEPATH}")
    print("Or install snakeviz for visual profiling:")
    print("  pip install snakeviz")
    print(f"  snakeviz {PROFILING_FILEPATH}")
