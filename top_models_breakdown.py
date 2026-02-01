# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
from utils import estimate_model_size, smooth_data

OMIT_KEYWORD = "405b"

# Load data
print("Loading data-15m.json...")
with open('data-15m.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']

# Data structures to hold per-timestamp model activity
# timestamp -> { repo_id: request_count }
window_data = defaultdict(lambda: defaultdict(int))

# Track total requests per model (for ranking)
total_requests_per_model = defaultdict(int)

# Cache for model sizes
model_size_cache = {}

def get_repo_id(model_key):
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        return model_key[start:end]
    return None

print("Processing frames and estimating model sizes...")
for frame in frames:
    labels = frame['schema']['fields'][1].get('labels', {})
    model_key = labels.get('model_key', 'unknown')
    repo_id = get_repo_id(model_key)
    
    if OMIT_KEYWORD in repo_id.lower():
        print(f"  Skipping {repo_id} because it contains {OMIT_KEYWORD}")
        continue

    if not repo_id:
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
            total_requests_per_model[repo_id] += count

# Rank models by total requests
sorted_models = sorted(total_requests_per_model.items(), key=lambda x: x[1], reverse=True)
model_rank = {repo_id: rank for rank, (repo_id, _) in enumerate(sorted_models)}
all_models = [repo_id for repo_id, _ in sorted_models]

print(f"\nTotal unique models: {len(all_models)}")
print("Top 10 models by request count:")
for i, (repo_id, count) in enumerate(sorted_models[:10]):
    print(f"  {i+1}. {repo_id}: {count:,} requests, {model_size_cache.get(repo_id, 0):.2f} GB")

# Sort timestamps
sorted_ts = sorted(window_data.keys())
plot_times = [datetime.fromtimestamp(ts / 1000.0) for ts in sorted_ts]

# 6-hour rolling window (24 slots of 15-min)
ROLLING_WINDOW = 24

# Top N configurations to compare
top_n_configs = [1, 2, 4, 8, 16, len(all_models)]
config_labels = ['Top 1', 'Top 2', 'Top 4', 'Top 8', 'Top 16', f'All ({len(all_models)})']
config_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

def compute_rolling_metrics_top_n(sorted_ts, window_data, model_size_cache, model_rank, top_n, window_size):
    """
    Compute rolling metrics but only including the top N models.
    """
    unique_models = []
    total_memory = []
    total_requests = []
    
    for i, ts in enumerate(sorted_ts):
        start_idx = max(0, i - window_size + 1)
        window_ts_list = sorted_ts[start_idx:i + 1]
        
        models_in_window = set()
        requests_in_window = 0
        
        for wts in window_ts_list:
            for repo_id, count in window_data[wts].items():
                # Only include if model is in top N
                if model_rank.get(repo_id, float('inf')) < top_n:
                    models_in_window.add(repo_id)
                    requests_in_window += count
        
        memory = sum(model_size_cache.get(m, 0) for m in models_in_window)
        
        unique_models.append(len(models_in_window))
        total_memory.append(memory)
        total_requests.append(requests_in_window)
    
    return unique_models, total_memory, total_requests

print("\nComputing 6h rolling metrics for different top-N configurations...")

# Precompute metrics for each top-N config
metrics_by_config = {}
for top_n, label in zip(top_n_configs, config_labels):
    print(f"  Computing {label}...")
    metrics_by_config[label] = compute_rolling_metrics_top_n(
        sorted_ts, window_data, model_size_cache, model_rank, top_n, ROLLING_WINDOW
    )

print("Generating plot...")

# Create 3-subplot figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Plot 1: Total Memory Usage
for label, color in zip(config_labels, config_colors):
    _, memory, _ = metrics_by_config[label]
    ax1.plot(plot_times, memory, color=color, linewidth=0.8, alpha=0.2)
    smoothed = smooth_data(memory)
    ax1.plot(plot_times, smoothed, color=color, linewidth=2, label=label)

ax1.set_ylabel('Total Memory (GB)')
ax1.set_title('Memory Required (6h TTL) - By Top N Models')
ax1.legend(loc='upper left', title='Models Included')
ax1.grid(True, alpha=0.3)

# Plot 2: Total Requests
for label, color in zip(config_labels, config_colors):
    _, _, requests = metrics_by_config[label]
    ax2.plot(plot_times, requests, color=color, linewidth=0.8, alpha=0.2)
    smoothed = smooth_data(requests)
    ax2.plot(plot_times, smoothed, color=color, linewidth=2, label=label)

ax2.set_ylabel('Total Requests')
ax2.set_title('Total Requests (6h Rolling) - By Top N Models')
ax2.legend(loc='upper left', title='Models Included')
ax2.grid(True, alpha=0.3)

# Plot 3: Unique Active Models
for label, color in zip(config_labels, config_colors):
    unique, _, _ = metrics_by_config[label]
    ax3.plot(plot_times, unique, color=color, linewidth=0.8, alpha=0.2)
    smoothed = smooth_data(unique)
    ax3.plot(plot_times, smoothed, color=color, linewidth=2, label=label)

ax3.set_ylabel('Unique Models')
ax3.set_title('Unique Models Active (6h Rolling) - By Top N Models')
ax3.set_xlabel('Time')
ax3.legend(loc='upper left', title='Models Included')
ax3.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY: 6h Rolling Window by Top N Models")
print("="*60)
for label in config_labels:
    unique, memory, requests = metrics_by_config[label]
    print(f"\n{label}:")
    print(f"  Unique models - avg: {np.mean(unique):.1f}, max: {max(unique)}")
    print(f"  Memory (GB)   - avg: {np.mean(memory):.1f}, max: {max(memory):.1f}, p95: {np.percentile(memory, 95):.1f}")
    print(f"  Requests      - avg: {np.mean(requests):.0f}, max: {max(requests)}, p95: {np.percentile(requests, 95):.0f}")

# Show coverage stats
print("\n" + "="*60)
print("COVERAGE: What % of requests do the top N models handle?")
print("="*60)
total_all = sum(total_requests_per_model.values())
cumulative = 0
for i, (repo_id, count) in enumerate(sorted_models):
    cumulative += count
    if i + 1 in [1, 2, 4, 8, 16, 32]:
        print(f"  Top {i+1}: {cumulative/total_all*100:.1f}% of all requests")
