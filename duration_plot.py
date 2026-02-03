# %%

import json

import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('duration.json', 'r') as f:
    data = json.load(f)

# Extract values from the data structure
values = data[0]['data']['values']
model_keys = values[0]
runtime_seconds = values[1]
percentiles = values[2]

# Parse model names and organize by model
model_data = {}

for model_key, runtime, percentile in zip(model_keys, runtime_seconds, percentiles):
    # Extract repo_id from the model_key string
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        model_name = model_key[start:end].split('/')[-1]
    else:
        model_name = model_key
    
    if model_name not in model_data:
        model_data[model_name] = {}
    model_data[model_name][percentile] = runtime

# Convert to arrays for plotting
models = list(model_data.keys())
p50_values = np.array([model_data[m].get('p50', 0) for m in models])
p90_values = np.array([model_data[m].get('p90', 0) for m in models])
p99_values = np.array([model_data[m].get('p99', 0) for m in models])

# Sort by p99 (total runtime) descending
sort_indices = np.argsort(p99_values)[::-1]
models = [models[i] for i in sort_indices]
p50_values = p50_values[sort_indices]
p90_values = p90_values[sort_indices]
p99_values = p99_values[sort_indices]

# Create the dot plot
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(models))

# Plot connecting lines (range from p50 to p99)
for i in range(len(models)):
    ax.plot([x[i], x[i]], [p50_values[i], p99_values[i]], 
            color='gray', linewidth=1.5, alpha=0.5, zorder=1)

# Plot dots for each percentile
ax.scatter(x, p50_values, s=60, color='#2ecc71', label='p50', zorder=2)
ax.scatter(x, p90_values, s=60, color='#f39c12', label='p90', zorder=2)
ax.scatter(x, p99_values, s=60, color='#e74c3c', label='p99', zorder=2)

# Formatting
ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Runtime (seconds)', fontsize=11)
ax.set_title('Request Runtime Percentiles by Model', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
ax.legend(loc='upper right')

# Log scale for y-axis with human-readable ticks
ax.set_yscale('log')

# Custom tick locations and labels
tick_values = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120]
tick_labels = ['0.1s', '0.25s', '0.5s', '1s', '2s', '5s', '10s', '30s', '1 min', '2 min']
ax.set_yticks(tick_values)
ax.set_yticklabels(tick_labels)

# Add gridlines for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# %%
# Print summary statistics
print("\n" + "=" * 60)
print("RUNTIME PERCENTILES SUMMARY")
print("=" * 60)
print(f"\nTotal models: {len(models)}")
print("\nTop 5 slowest models (by p99):")
for i, model in enumerate(models[:5]):
    print(f"  {i+1}. {model}: p50={p50_values[i]:.2f}s, p90={p90_values[i]:.2f}s, p99={p99_values[i]:.2f}s")

print("\nFastest 5 models (by p99):")
for i, model in enumerate(models[-5:][::-1]):
    idx = len(models) - 5 + (4 - i)
    print(f"  {5-i}. {model}: p50={p50_values[idx]:.2f}s, p90={p90_values[idx]:.2f}s, p99={p99_values[idx]:.2f}s")
