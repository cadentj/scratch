# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Aggregation window in minutes (raw data is 15m intervals)
# Examples: 15 (raw), 60 (1h), 360 (6h), 1440 (24h)
WINDOW_MINUTES = 120

# =============================================================================

# Load data
print("Loading data-15m.json...")
with open('data-15m.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']

# Data structures
# timestamp -> { repo_id: request_count }
window_data = defaultdict(lambda: defaultdict(int))

def get_repo_id(model_key):
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        return model_key[start:end]
    return None

print("Processing frames...")
for frame in frames:
    labels = frame['schema']['fields'][1].get('labels', {})
    model_key = labels.get('model_key', 'unknown')
    repo_id = get_repo_id(model_key)
    
    if not repo_id:
        continue
    
    timestamps = frame['data']['values'][0]
    counts = frame['data']['values'][1]
    
    for ts, count in zip(timestamps, counts):
        window_data[ts][repo_id] += count

# Sort timestamps globally
sorted_ts = sorted(window_data.keys())
print(f"Total time windows: {len(sorted_ts)}")

# %%
# =============================================================================
# AGGREGATE INTO LARGER WINDOWS (if needed)
# =============================================================================

RAW_WINDOW_MINUTES = 15
slots_per_window = max(1, WINDOW_MINUTES // RAW_WINDOW_MINUTES)

if slots_per_window > 1:
    print(f"\nAggregating {RAW_WINDOW_MINUTES}m windows into {WINDOW_MINUTES}m windows...")
    # Group timestamps into larger buckets
    window_ms = WINDOW_MINUTES * 60 * 1000
    aggregated_data = defaultdict(lambda: defaultdict(int))
    
    for ts in sorted_ts:
        bucket_ts = (ts // window_ms) * window_ms
        for repo_id, count in window_data[ts].items():
            aggregated_data[bucket_ts][repo_id] += count
    
    sorted_ts = sorted(aggregated_data.keys())
    window_data = aggregated_data
    print(f"Aggregated into {len(sorted_ts)} windows")

# %%
# =============================================================================
# COMPUTE CV FOR EACH MODEL
# =============================================================================

print(f"\nComputing per-model CV (window={WINDOW_MINUTES}m)...")

# Create full time series for each model (with zeros for inactive windows)
model_full_series = {}
for repo_id in set(repo_id for ts in sorted_ts for repo_id in window_data[ts].keys()):
    series = []
    for ts in sorted_ts:
        series.append(window_data[ts].get(repo_id, 0))
    model_full_series[repo_id] = series

# Compute CV for each model
model_metrics = {}
for repo_id, series in model_full_series.items():
    counts = np.array(series)
    total = np.sum(counts)
    
    if len(counts) < 2 or total == 0:
        continue
    
    mean = np.mean(counts)
    if mean == 0:
        continue
    
    std = np.std(counts)
    cv = std / mean
    
    if total >= 10:  # Only models with at least 10 requests
        model_metrics[repo_id] = {'cv': cv, 'total': total}

print(f"Models with sufficient data: {len(model_metrics)}")

# Sort by total requests (descending)
sorted_by_requests = sorted(model_metrics.items(), key=lambda x: x[1]['total'], reverse=True)

# %%
# =============================================================================
# PER-MODEL CV BAR CHART (sorted by total requests)
# =============================================================================

# Get model names and CV values, sorted by total requests (descending)
model_names = [repo_id.split('/')[-1] for repo_id, _ in sorted_by_requests]
cv_vals = [m['cv'] for _, m in sorted_by_requests]
totals = [m['total'] for _, m in sorted_by_requests]

# Limit to top N for readability
TOP_N = 30
model_names = model_names[:TOP_N]
cv_vals = cv_vals[:TOP_N]
totals = totals[:TOP_N]

fig, ax = plt.subplots(figsize=(12, 8))

# Color bars by CV value (higher = more bursty = redder)
colors = plt.colormaps['RdYlGn_r'](np.array(cv_vals) / max(cv_vals))

bars = ax.barh(range(len(model_names)), cv_vals, color=colors)

# Add reference line at CV=1
ax.axvline(x=1, color='black', linestyle='--', alpha=0.7, label='CV=1 (std=mean)')

ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names)
ax.invert_yaxis()  # Highest requests at top
ax.set_xlabel('Coefficient of Variation (CV = σ/μ)')
ax.set_title(f'Per-Model Usage Variability (Top {TOP_N} by Request Volume, {WINDOW_MINUTES}m windows)\nHigher CV = More Bursty')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

# Add request counts as secondary labels
for i, (cv, total) in enumerate(zip(cv_vals, totals)):
    ax.text(cv + 0.05, i, f'{total:,} req', va='center', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()
