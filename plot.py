# %%

import json

import matplotlib.pyplot as plt
import numpy as np

from estimate import estimate_model_size

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

frames = data['response']['results']['A']['frames']

# Extract model usage
model_usage = {}

for frame in frames:
    labels = frame['schema']['fields'][1].get('labels', {})
    model_key = labels.get('model_key', 'unknown')
    
    # Parse model name (extract repo_id from the JSON-like string)
    if 'repo_id' in model_key:
        start = model_key.find('"repo_id": "') + len('"repo_id": "')
        end = model_key.find('"', start)
        model_name = model_key[start:end].split('/')[-1]  # Just the model name
    else:
        model_name = model_key
    
    timestamps = frame['data']['values'][0]
    counts = frame['data']['values'][1]
    
    total = sum(counts)
    
    # Aggregate if model appears multiple times (due to different configs)
    if model_name in model_usage:
        model_usage[model_name] += total
    else:
        model_usage[model_name] = total

# Sort by usage descending
sorted_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)
names = [m[0] for m in sorted_models]
counts = [m[1] for m in sorted_models]

# Calculate cumulative percentage
total_requests = sum(counts)
cumulative = np.cumsum(counts) / total_requests * 100

# Compute memory usage for each model (in same order as sorted_models)
print("Estimating model sizes...")
model_sizes = []
for model_name in names:
    # Find the full repo_id for this model from the original data
    full_repo_id = None
    for frame in frames:
        labels = frame['schema']['fields'][1].get('labels', {})
        model_key = labels.get('model_key', 'unknown')
        if 'repo_id' in model_key:
            start = model_key.find('"repo_id": "') + len('"repo_id": "')
            end = model_key.find('"', start)
            repo_id = model_key[start:end]
            if repo_id.split('/')[-1] == model_name:
                full_repo_id = repo_id
                break
    
    if full_repo_id:
        try:
            size_bytes = estimate_model_size(full_repo_id)
            print(f"  {model_name}: {size_bytes / 1e9:.2f} GB")
        except Exception as e:
            print(f"  {model_name}: Failed to estimate ({e})")
            size_bytes = 0
    else:
        print(f"  {model_name}: No repo_id found")
        size_bytes = 0
    model_sizes.append(size_bytes)

# Calculate cumulative memory usage
cumulative_memory = np.cumsum(model_sizes)
cumulative_memory_gb = cumulative_memory / 1e9

# Create figure with subplots (top + bottom only)
total_memory_gb = cumulative_memory_gb[-1] if len(cumulative_memory_gb) > 0 else 0
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Bar chart with Pareto line
ax1 = axes[0]
ax1_twin = ax1.twinx()

bars = ax1.bar(range(len(names)), counts, color='steelblue', alpha=0.7)
ax1_twin.plot(range(len(names)), cumulative, 'ro-', linewidth=2, markersize=4)

# Add threshold lines
ax1_twin.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
ax1_twin.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')

# Find where thresholds are crossed
models_for_50 = np.searchsorted(cumulative, 50) + 1
models_for_80 = np.searchsorted(cumulative, 80) + 1
models_for_90 = np.searchsorted(cumulative, 90) + 1
models_for_99 = np.searchsorted(cumulative, 99) + 1

ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Total Requests', fontsize=11)
ax1_twin.set_ylabel('Cumulative %', fontsize=11)
ax1.set_title(f'Model Usage\n'
              f'Top {models_for_80} models = 80% of traffic | '
              f'Top {models_for_90} models = 90% of traffic', fontsize=12)
ax1_twin.legend(loc='center right')
ax1.set_xlim(-0.5, len(names) - 0.5)

# Plot 2: Cumulative memory with request threshold markers
ax3 = axes[1]

ax3.plot(range(len(names)), cumulative_memory_gb, 'g-', linewidth=2)
ax3.fill_between(range(len(names)), cumulative_memory_gb, alpha=0.3, color='green')

# Get memory at each request threshold
mem_50 = cumulative_memory_gb[models_for_50 - 1]
mem_80 = cumulative_memory_gb[models_for_80 - 1]
mem_90 = cumulative_memory_gb[models_for_90 - 1]
mem_99 = cumulative_memory_gb[min(models_for_99 - 1, len(cumulative_memory_gb) - 1)]

# Plot threshold lines and markers
thresholds = [
    (models_for_50 - 1, mem_50, '50%', 'blue'),
    (models_for_80 - 1, mem_80, '80%', 'orange'),
    (models_for_90 - 1, mem_90, '90%', 'red'),
    (models_for_99 - 1, mem_99, '99%', 'purple'),
]

for idx, mem, label, color in thresholds:
    ax3.axvline(x=idx, color=color, linestyle='--', alpha=0.7)
    ax3.axhline(y=mem, color=color, linestyle=':', alpha=0.5)
    ax3.scatter([idx], [mem], color=color, s=100, zorder=5)
    ax3.annotate(f'{label}: {mem:.1f} GB', 
                 xy=(idx, mem), 
                 xytext=(idx + 1, mem + total_memory_gb * 0.05),
                 fontsize=9, color=color, fontweight='bold', rotation=45)

ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Cumulative Memory (GB)', fontsize=11)
ax3.set_xlabel('Models (ordered by request volume)', fontsize=11)
ax3.set_title('Cumulative Memory Required for Request Coverage Thresholds', fontsize=12)
ax3.set_xlim(-0.5, len(names) - 0.5)
ax3.set_ylim(0, total_memory_gb * 1.1)

plt.tight_layout()
plt.savefig('model_usage_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary stats
print("\n" + "="*60)
print("SUMMARY: Long Tail Analysis")
print("="*60)
print(f"\nTotal models tracked: {len(names)}")
print(f"Total requests: {total_requests:,}")
print(f"\nConcentration:")
print(f"  - Top 1 model:  {cumulative[0]:.1f}% of traffic")
print(f"  - Top 3 models: {cumulative[2]:.1f}% of traffic")
print(f"  - Top 5 models: {cumulative[4]:.1f}% of traffic")
print(f"  - Models needed for 80%: {models_for_80}")
print(f"  - Models needed for 90%: {models_for_90}")
print(f"\nMemory Usage for Request Coverage:")
print(f"  - 50% of requests ({models_for_50} models): {mem_50:.1f} GB")
print(f"  - 80% of requests ({models_for_80} models): {mem_80:.1f} GB")
print(f"  - 90% of requests ({models_for_90} models): {mem_90:.1f} GB")
print(f"  - 99% of requests ({models_for_99} models): {mem_99:.1f} GB")
print(f"  - Total (all {len(names)} models): {total_memory_gb:.1f} GB")