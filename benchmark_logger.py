import csv
import os
from datetime import datetime

BENCHMARK_FILE = "benchmarks.csv"

HEADERS = [
    "timestamp",
    "query",
    "intent",
    "category",
    "tool_generation_time_sec",
    "parameter_extraction_success",
    "tool_execution_success",
    "end_to_end_latency_sec",
    "error_message"
]

def log_benchmark(
    query: str,
    intent: str,
    category: str = "N/A",
    tool_generation_time: float = None,
    parameter_extraction_success: str = "N/A",
    tool_execution_success: str = "N/A",
    end_to_end_latency: float = 0.0,
    error_message: str = ""
):
    """Logs dynamic chatbot execution metrics to a CSV file."""
    file_exists = os.path.exists(BENCHMARK_FILE)
    
    # Using 'utf-8' encoding and thread-safe write model
    with open(BENCHMARK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADERS)
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            intent,
            category,
            f"{tool_generation_time:.4f}" if tool_generation_time is not None else "N/A",
            parameter_extraction_success,
            tool_execution_success,
            f"{end_to_end_latency:.4f}",
            str(error_message) if error_message else ""
        ])
