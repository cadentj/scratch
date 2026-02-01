# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
from utils.estimate import estimate_model_size

# ============================================================
# CONFIGURATION
# ============================================================
# Rolling window size for cache TTL simulation (in 15-min slots)
# 6 hours = 24 slots, 1 hour = 4 slots, 24 hours = 96 slots
WINDOW_SIZE = 24  # 6 hours
OMIT_KEYWORD = "405b"
# ============================================================

# Load data
print("Loading data-15m.json...")
with open('data-15m.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']

# Data structures to hold per-timestamp model activity
# timestamp -> set of repo_ids active in that 15-min slot
slot_models = defaultdict(set)

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
            slot_models[ts].add(repo_id)

# Sort timestamps
sorted_ts = sorted(slot_models.keys())
plot_times = [datetime.fromtimestamp(ts / 1000.0) for ts in sorted_ts]

window_hours = WINDOW_SIZE * 15 / 60
print(f"\nComputing hotswap metrics with {window_hours:.1f}h rolling window...")

def get_rolling_window_models(sorted_ts, slot_models, end_idx, window_size):
    """Get all unique models active in the rolling window ending at end_idx."""
    start_idx = max(0, end_idx - window_size + 1)
    models = set()
    for i in range(start_idx, end_idx + 1):
        models.update(slot_models[sorted_ts[i]])
    return models

# Compute metrics for each consecutive rolling window pair
models_entering = []  # New models this window (not in previous)
models_exiting = []   # Models that left (were in previous, not in current)
memory_entering = []  # GB of new models
memory_exiting = []   # GB of exiting models
memory_churn = []     # Net memory change (entering - exiting)
overlap_ratio = []    # % of previous window's models still present

for i in range(len(sorted_ts)):
    current_models = get_rolling_window_models(sorted_ts, slot_models, i, WINDOW_SIZE)
    
    if i == 0:
        prev_models = set()
    else:
        prev_models = get_rolling_window_models(sorted_ts, slot_models, i - 1, WINDOW_SIZE)
    
    # Models entering (in current but not in previous)
    entering = current_models - prev_models
    # Models exiting (in previous but not in current)
    exiting = prev_models - current_models
    # Models that stayed
    stayed = current_models & prev_models
    
    models_entering.append(len(entering))
    models_exiting.append(len(exiting))
    
    mem_in = sum(model_size_cache.get(m, 0) for m in entering)
    mem_out = sum(model_size_cache.get(m, 0) for m in exiting)
    memory_entering.append(mem_in)
    memory_exiting.append(mem_out)
    memory_churn.append(mem_in - mem_out)
    
    # Overlap ratio: what % of previous models are still here?
    if len(prev_models) > 0:
        overlap_ratio.append(len(stayed) / len(prev_models) * 100)
    else:
        overlap_ratio.append(100.0)

print("Generating plot...")

# Create 3-subplot figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

window_label = f"{window_hours:.0f}h" if window_hours >= 1 else f"{WINDOW_SIZE * 15}min"

# Plot 1: Swap Events (models entering and exiting)
ax1.plot(plot_times, models_entering, color='tab:green', linewidth=1, label='Models entering')
ax1.plot(plot_times, models_exiting, color='tab:red', linewidth=1, label='Models exiting')

ax1.set_ylabel('Model Count')
ax1.set_title(f'Swap Events: Models Entering/Exiting Cache ({window_label} TTL)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=0.5)

# Plot 2: Memory Churn (GB entering, exiting, net)
ax2.plot(plot_times, memory_entering, color='tab:green', linewidth=1, label='Memory entering (GB)')
ax2.plot(plot_times, memory_exiting, color='tab:red', linewidth=1, label='Memory exiting (GB)')
ax2.plot(plot_times, memory_churn, color='tab:blue', linewidth=1, linestyle='--', label='Net churn (GB)')

ax2.set_ylabel('Memory (GB)')
ax2.set_title(f'Memory Churn: GB of Models Swapping In/Out ({window_label} TTL)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)

# Plot 3: Overlap Ratio
ax3.plot(plot_times, overlap_ratio, color='tab:purple', linewidth=1, label='Overlap ratio')

ax3.set_ylabel('Overlap (%)')
ax3.set_title(f'Cache Stability: % of Previous {window_label} Window Models Still Active')
ax3.set_xlabel('Time')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 105)
ax3.axhline(y=100, color='black', linewidth=0.5, linestyle=':')

plt.xticks(rotation=45)
plt.tight_layout()

output_file = 'hotswap_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_file}")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print(f"SUMMARY: Hotswap Analysis ({window_label} TTL)")
print("="*60)

print(f"\nSwap Events (per step, comparing consecutive {window_label} windows):")
print(f"  Models entering - avg: {np.mean(models_entering):.2f}, max: {max(models_entering)}, p95: {np.percentile(models_entering, 95):.0f}")
print(f"  Models exiting  - avg: {np.mean(models_exiting):.2f}, max: {max(models_exiting)}, p95: {np.percentile(models_exiting, 95):.0f}")

print("\nMemory Churn (per step):")
print(f"  Memory entering - avg: {np.mean(memory_entering):.2f} GB, max: {max(memory_entering):.1f} GB, p95: {np.percentile(memory_entering, 95):.1f} GB")
print(f"  Memory exiting  - avg: {np.mean(memory_exiting):.2f} GB, max: {max(memory_exiting):.1f} GB, p95: {np.percentile(memory_exiting, 95):.1f} GB")
print(f"  Net churn       - avg: {np.mean(memory_churn):.2f} GB, range: [{min(memory_churn):.1f}, {max(memory_churn):.1f}] GB")

print("\nCache Stability:")
print(f"  Overlap ratio - avg: {np.mean(overlap_ratio):.1f}%, min: {min(overlap_ratio):.1f}%, p5: {np.percentile(overlap_ratio, 5):.1f}%")

# Count windows with no swaps
no_swap_windows = sum(1 for e, x in zip(models_entering, models_exiting) if e == 0 and x == 0)
print(f"  Steps with no swaps: {no_swap_windows}/{len(sorted_ts)} ({no_swap_windows/len(sorted_ts)*100:.1f}%)")

# %%
