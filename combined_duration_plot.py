# %%

import json

import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('combined_duration.json', 'r') as f:
    data = json.load(f)

# Extract values
values = data[0]['data']['values']
runtime_seconds = values[0]
percentile_labels = values[1]

# Convert percentile labels to numeric (e.g., "p50" -> 50)
percentiles = [int(p[1:]) for p in percentile_labels]

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(percentiles, runtime_seconds, s=80, color='#3498db', zorder=2)
ax.plot(percentiles, runtime_seconds, color='#3498db', alpha=0.5, linewidth=1.5, zorder=1)

# Formatting
ax.set_xlabel('Percentile', fontsize=11)
ax.set_ylabel('Runtime', fontsize=11)
ax.set_title('Request Runtime Distribution (All Models Combined)', fontsize=12)

# X-axis ticks at each percentile
ax.set_xticks(percentiles)
ax.set_xticklabels([f'p{p}' for p in percentiles], rotation=45, ha='right', fontsize=9)

# Log scale for y-axis with human-readable ticks
ax.set_yscale('log')

tick_values = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200]
tick_labels = ['0.1s', '0.25s', '0.5s', '1s', '2s', '5s', '10s', '30s', '1m', '2m', '5m', '10m', '30m', '1h', '2h']

# Filter ticks to data range
y_min, y_max = min(runtime_seconds) * 0.8, max(runtime_seconds) * 1.2
filtered = [(v, l) for v, l in zip(tick_values, tick_labels) if y_min <= v <= y_max]
if filtered:
    tick_values, tick_labels = zip(*filtered)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)

ax.set_ylim(y_min, y_max)

# Add gridlines
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# %%
# Print summary
print("\n" + "=" * 50)
print("RUNTIME PERCENTILES SUMMARY")
print("=" * 50)
for p, r in zip(percentile_labels, runtime_seconds):
    if r < 60:
        fmt = f"{r:.2f}s"
    elif r < 3600:
        fmt = f"{r/60:.1f} min"
    else:
        fmt = f"{r/3600:.1f} hr"
    print(f"  {p:>4}: {fmt}")
