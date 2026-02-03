# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum number of requests for a model to be included
MIN_REQUESTS = 50

# Number of top models to display (by request count)
TOP_N = 12

# =============================================================================

# Load data
print("Loading data-15m.json...")
with open('data-15m.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']


def get_repo_id(model_key):
    """Extract repo_id from model_key string."""
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        return model_key[start:end]
    return None


# Collect all timestamps (with their counts) for each model
print("Processing frames...")
model_timestamps = defaultdict(list)

for frame in frames:
    labels = frame['schema']['fields'][1].get('labels', {})
    model_key = labels.get('model_key', 'unknown')
    repo_id = get_repo_id(model_key)
    
    if not repo_id:
        continue
    
    timestamps = frame['data']['values'][0]
    counts = frame['data']['values'][1]
    
    # Store each timestamp (repeated by count for weighting, or just once)
    for ts, count in zip(timestamps, counts):
        # Each window where activity occurred represents a "request period"
        model_timestamps[repo_id].append(ts)

# Sort timestamps and compute gaps for each model
print("Computing time gaps...")
model_gaps = {}
model_totals = {}

for repo_id, timestamps in model_timestamps.items():
    if len(timestamps) < 2:
        continue
    
    sorted_ts = sorted(timestamps)
    # Calculate gaps in hours
    gaps = [(sorted_ts[i+1] - sorted_ts[i]) / (1000 * 60 * 60) 
            for i in range(len(sorted_ts) - 1)]
    
    total_windows = len(timestamps)
    if total_windows >= MIN_REQUESTS:
        model_gaps[repo_id] = gaps
        model_totals[repo_id] = total_windows

print(f"Models with sufficient data: {len(model_gaps)}")

# Sort models by total activity windows (descending)
sorted_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

# %%
# =============================================================================
# PLOT: LOG-SCALE HISTOGRAMS FOR EACH MODEL
# =============================================================================

n_models = len(sorted_models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
axes = axes.flatten()

for idx, (repo_id, total) in enumerate(sorted_models):
    ax = axes[idx]
    gaps = model_gaps[repo_id]
    
    # Use log-scale bins for better visualization of the distribution
    gaps_positive = [g for g in gaps if g > 0]
    if len(gaps_positive) < 2:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(repo_id.split('/')[-1])
        continue
    
    log_gaps = np.log10(gaps_positive)
    
    # Create histogram with log-scale bins
    bins = np.logspace(np.floor(log_gaps.min()), np.ceil(log_gaps.max()), 30)
    
    ax.hist(gaps_positive, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Time Gap (hours)')
    ax.set_ylabel('Frequency')
    
    model_name = repo_id.split('/')[-1]
    median_gap = np.median(gaps_positive)
    ax.set_title(f'{model_name}\n({total} windows, median={median_gap:.1f}h)')
    ax.axvline(median_gap, color='red', linestyle='--', alpha=0.7, label=f'Median')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(sorted_models), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Distribution of Time Gaps Between Consecutive Activity Windows (Log Scale)\n'
             'Bimodal = Bursty Usage, Unimodal = Consistent Usage', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# PLOT: KERNEL DENSITY ESTIMATES (KDE) COMPARISON
# =============================================================================

from scipy import stats

fig, ax = plt.subplots(figsize=(12, 6))

# Use a colormap for different models
colors = plt.colormaps['tab10']

for idx, (repo_id, total) in enumerate(sorted_models):  # Top 8 for clarity
    gaps = model_gaps[repo_id]
    gaps_positive = [g for g in gaps if g > 0]
    
    if len(gaps_positive) < 10:
        continue
    
    # Work in log space for KDE
    log_gaps = np.log10(gaps_positive)
    
    # Compute KDE
    kde = stats.gaussian_kde(log_gaps)
    x_range = np.linspace(log_gaps.min() - 0.5, log_gaps.max() + 0.5, 200)
    density = kde(x_range)
    
    model_name = repo_id.split('/')[-1]
    ax.plot(10**x_range, density, label=f'{model_name} ({total})', 
            color=colors(idx), linewidth=2, alpha=0.8)

ax.set_xscale('log')
ax.set_xlabel('Time Gap (hours)', fontsize=11)
ax.set_ylabel('Density (log-scale)', fontsize=11)
ax.set_title('Kernel Density Estimates of Time Gaps Between Activity\n'
             '(Bimodal peaks suggest bursty usage patterns)', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Add reference lines
ax.axvline(0.25, color='gray', linestyle=':', alpha=0.5, label='15 min')
ax.axvline(1, color='gray', linestyle=':', alpha=0.5)
ax.axvline(24, color='gray', linestyle=':', alpha=0.5)

# Add text annotations for reference lines
ax.text(0.25, ax.get_ylim()[1]*0.95, '15m', fontsize=8, alpha=0.6, ha='center')
ax.text(1, ax.get_ylim()[1]*0.95, '1h', fontsize=8, alpha=0.6, ha='center')
ax.text(24, ax.get_ylim()[1]*0.95, '24h', fontsize=8, alpha=0.6, ha='center')

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# SUMMARY STATISTICS TABLE
# =============================================================================

print("\n" + "="*80)
print("TIME GAP DISTRIBUTION SUMMARY (Top Models by Activity)")
print("="*80)
print(f"{'Model':<35} {'Windows':>8} {'Median':>10} {'Mean':>10} {'Std':>10} {'CV':>8}")
print("-"*80)

for repo_id, total in sorted_models:
    gaps = model_gaps[repo_id]
    gaps_positive = [g for g in gaps if g > 0]
    
    if len(gaps_positive) < 2:
        continue
    
    median_gap = np.median(gaps_positive)
    mean_gap = np.mean(gaps_positive)
    std_gap = np.std(gaps_positive)
    cv = std_gap / mean_gap if mean_gap > 0 else 0
    
    model_name = repo_id.split('/')[-1][:34]
    print(f"{model_name:<35} {total:>8} {median_gap:>9.1f}h {mean_gap:>9.1f}h {std_gap:>9.1f}h {cv:>7.2f}")

print("="*80)
print("CV (Coefficient of Variation) > 1 suggests high variability/burstiness")
print("Large gap between median and mean suggests skewed distribution (common in bursty usage)")
