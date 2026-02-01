# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
from estimate import estimate_model_size

# Load data
print("Loading data-15m.json...")
with open('data-15m.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']

# Data structures to hold per-timestamp model activity
# timestamp -> { repo_id: request_count }
window_data = defaultdict(lambda: defaultdict(int))

# Cache for model sizes to avoid redundant HF API calls/estimation
model_size_cache = {}

def get_repo_id(model_key):
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        return model_key[start:end]
    return None

print("Processing frames and estimating model sizes (excluding 405b models)...")
for frame in frames:
    labels = frame['schema']['fields'][1].get('labels', {})
    model_key = labels.get('model_key', 'unknown')
    repo_id = get_repo_id(model_key)
    
    if not repo_id:
        continue
    
    # Skip 405b models
    if '405b' in repo_id.lower():
        continue

    # Get or estimate model size
    if repo_id not in model_size_cache:
        try:
            size_gb = estimate_model_size(repo_id) / 1e9
            model_size_cache[repo_id] = size_gb
            print(f"  {repo_id}: {size_gb:.2f} GB")
        except Exception as e:
            print(f"  Warning: Could not estimate size for {repo_id}: {e}")
            model_size_cache[repo_id] = 0.0
    
    timestamps = frame['data']['values'][0]
    counts = frame['data']['values'][1]
    
    for ts, count in zip(timestamps, counts):
        if count > 0:
            window_data[ts][repo_id] += count

# Sort timestamps
sorted_ts = sorted(window_data.keys())
plot_times = [datetime.fromtimestamp(ts / 1000.0) for ts in sorted_ts]

# Define rolling windows (in number of 15-min slots)
# 15min = 1 slot, so: 1h=4, 6h=24, 24h=96
rolling_windows = {
    '15min': 1,
    '1h': 4,
    '6h': 24,
    '24h': 96,
}

# Colors for each window
window_colors = {
    '15min': 'tab:green',
    '1h': 'tab:blue',
    '6h': 'tab:orange',
    '24h': 'tab:red',
}

# Smoothing window size (2 hours = 8 slots of 15-min each)
SMOOTHING_WINDOW = 8

def smooth_data(data, window_size=SMOOTHING_WINDOW):
    """Apply a centered rolling mean to smooth the data."""
    data = np.array(data, dtype=float)
    kernel = np.ones(window_size) / window_size
    # Use 'same' mode and handle edges with valid data
    smoothed = np.convolve(data, kernel, mode='same')
    # Fix edge effects by using smaller windows at the edges
    for i in range(window_size // 2):
        left_window = i + 1
        smoothed[i] = np.mean(data[:left_window + window_size // 2])
        right_idx = len(data) - 1 - i
        smoothed[right_idx] = np.mean(data[right_idx - window_size // 2:])
    return smoothed

def compute_rolling_metrics(sorted_ts, window_data, model_size_cache, window_size):
    """
    For each timestamp, compute metrics over the rolling window ending at that timestamp.
    Returns: (unique_models_list, total_memory_list, total_requests_list)
    """
    unique_models = []
    total_memory = []
    total_requests = []
    
    for i, ts in enumerate(sorted_ts):
        # Get timestamps in the rolling window
        start_idx = max(0, i - window_size + 1)
        window_ts_list = sorted_ts[start_idx:i + 1]
        
        # Collect all unique models and sum requests in window
        models_in_window = set()
        requests_in_window = 0
        
        for wts in window_ts_list:
            for repo_id, count in window_data[wts].items():
                models_in_window.add(repo_id)
                requests_in_window += count
        
        # Calculate memory for unique models
        memory = sum(model_size_cache.get(m, 0) for m in models_in_window)
        
        unique_models.append(len(models_in_window))
        total_memory.append(memory)
        total_requests.append(requests_in_window)
    
    return unique_models, total_memory, total_requests

print(f"\nComputing rolling metrics for {len(sorted_ts)} windows...")

# Precompute all rolling metrics
rolling_metrics = {}
for window_name, window_size in rolling_windows.items():
    print(f"  Computing {window_name} rolling window...")
    rolling_metrics[window_name] = compute_rolling_metrics(
        sorted_ts, window_data, model_size_cache, window_size
    )

print("Generating plot...")

# Create 3-subplot figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Plot 1: Total Memory Usage (rolling unique models' memory)
for window_name in rolling_windows.keys():
    _, memory, _ = rolling_metrics[window_name]
    color = window_colors[window_name]
    # Raw data at low opacity
    ax1.plot(plot_times, memory, color=color, linewidth=0.8, alpha=0.2)
    # Smoothed line at full opacity
    smoothed = smooth_data(memory)
    ax1.plot(plot_times, smoothed, color=color, linewidth=2, label=window_name)

ax1.set_ylabel('Total Memory (GB)')
ax1.set_title('Memory Required to Cache All Unique Models (by TTL window, excluding 405b)')
ax1.legend(loc='upper left', title='Cache TTL')
ax1.grid(True, alpha=0.3)

# Plot 2: Total Requests (rolling sum)
for window_name in rolling_windows.keys():
    _, _, requests = rolling_metrics[window_name]
    color = window_colors[window_name]
    # Raw data at low opacity
    ax2.plot(plot_times, requests, color=color, linewidth=0.8, alpha=0.2)
    # Smoothed line at full opacity
    smoothed = smooth_data(requests)
    ax2.plot(plot_times, smoothed, color=color, linewidth=2, label=window_name)

ax2.set_ylabel('Total Requests')
ax2.set_title('Total Requests in Rolling Window (excluding 405b)')
ax2.legend(loc='upper left', title='Window')
ax2.grid(True, alpha=0.3)

# Plot 3: Unique Active Models (rolling count)
for window_name in rolling_windows.keys():
    unique, _, _ = rolling_metrics[window_name]
    color = window_colors[window_name]
    # Raw data at low opacity
    ax3.plot(plot_times, unique, color=color, linewidth=0.8, alpha=0.2)
    # Smoothed line at full opacity
    smoothed = smooth_data(unique)
    ax3.plot(plot_times, smoothed, color=color, linewidth=2, label=window_name)

ax3.set_ylabel('Unique Models')
ax3.set_title('Unique Models Seen in Rolling Window (excluding 405b)')
ax3.set_xlabel('Time')
ax3.legend(loc='upper left', title='Window')
ax3.grid(True, alpha=0.3)

# Improve x-axis formatting
plt.xticks(rotation=45)
plt.tight_layout()

output_file = 'window_stats_analysis_no_405b.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_file}")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY: Cache Memory Requirements (excluding 405b)")
print("="*60)
for window_name in rolling_windows.keys():
    unique, memory, requests = rolling_metrics[window_name]
    print(f"\n{window_name} TTL:")
    print(f"  Unique models - avg: {np.mean(unique):.1f}, max: {max(unique)}, p95: {np.percentile(unique, 95):.0f}")
    print(f"  Memory (GB)   - avg: {np.mean(memory):.1f}, max: {max(memory):.1f}, p95: {np.percentile(memory, 95):.1f}")
