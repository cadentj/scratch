# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load data
with open('data-15m.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']

# Extract per-model timeseries
model_timeseries = {}

for frame in frames:
    labels = frame['schema']['fields'][1].get('labels', {})
    model_key = labels.get('model_key', 'unknown')
    
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        model_name = model_key[start:end].split('/')[-1]
    else:
        model_name = model_key
    
    timestamps = frame['data']['values'][0]
    counts = frame['data']['values'][1]
    
    if model_name not in model_timeseries:
        model_timeseries[model_name] = {'timestamps': [], 'counts': []}
    
    model_timeseries[model_name]['timestamps'].extend(timestamps)
    model_timeseries[model_name]['counts'].extend(counts)

# Calculate totals and sort by usage
model_totals = {m: sum(d['counts']) for m, d in model_timeseries.items()}
sorted_models = sorted(model_totals.keys(), key=lambda x: model_totals[x], reverse=True)

# Get time range
all_ts = []
for d in model_timeseries.values():
    all_ts.extend(d['timestamps'])
min_ts, max_ts = min(all_ts), max(all_ts)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

n_models = len(sorted_models)

for i, model in enumerate(sorted_models):
    ts = model_timeseries[model]['timestamps']
    counts = model_timeseries[model]['counts']
    
    # Normalize timestamps to 0-1 range for plotting
    x_positions = [(t - min_ts) / (max_ts - min_ts) for t in ts]
    
    # Size dots by count (log scale for visibility)
    sizes = [3 + 15 * np.log1p(c) for c in counts]
    
    # Plot
    y_positions = [n_models - i - 1] * len(x_positions)
    ax.scatter(x_positions, y_positions, s=sizes, alpha=0.6, 
               color='steelblue', edgecolors='none')

# Y-axis labels with total counts
y_labels = [f"{m[:25]} ({model_totals[m]:,})" for m in sorted_models]
ax.set_yticks(range(n_models))
ax.set_yticklabels(reversed(y_labels), fontsize=8)

# X-axis as dates
n_ticks = 8
tick_positions = np.linspace(0, 1, n_ticks)
tick_timestamps = [min_ts + p * (max_ts - min_ts) for p in tick_positions]
tick_labels = [datetime.fromtimestamp(t/1000).strftime('%m/%d') for t in tick_timestamps]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=9)

ax.set_xlabel('Date', fontsize=11)
ax.set_title('Model Activity Timeline (dot size = request volume in window)\n'
             'Sorted by total usage (top = highest)', fontsize=12)

# Add gridlines for readability
ax.set_xlim(-0.02, 1.02)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add separator line between head and tail (top 5 vs rest)
ax.axhline(y=n_models - 5.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Add separator line between more frequently used tail models
ax.axhline(y=n_models - 16.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('session_view.png', dpi=150, bbox_inches='tight')
plt.show()

# Session detection: count clusters of activity (gap > 2 hours = new session)
print("\n" + "="*70)
print("SESSION ANALYSIS (gap > 2 hours = new session)")
print("="*70)
print(f"\n{'Model':<28} {'Total':>8} {'Sessions':>8} {'Avg/Sess':>10}")
print("-"*70)

session_gap_ms = 2 * 60 * 60 * 1000  # 2 hours in ms

for model in sorted_models:
    ts = sorted(model_timeseries[model]['timestamps'])
    total = model_totals[model]
    
    if len(ts) <= 1:
        sessions = 1
    else:
        sessions = 1
        for j in range(1, len(ts)):
            if ts[j] - ts[j-1] > session_gap_ms:
                sessions += 1
    
    avg_per_session = total / sessions
    print(f"{model[:28]:<28} {total:>8,} {sessions:>8} {avg_per_session:>10,.1f}")