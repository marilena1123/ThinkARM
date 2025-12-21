import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import tiktoken

# base_path = "data/sequence_l/model"
label_dir = "data/label"
models = ['deepseekR1', 'Phi4R', 'Qwen3_32B', 'QwQ32B', 'DSQwen32B']
curve_data = {}

tokenizer = tiktoken.get_encoding("cl100k_base")

def get_token_length(sentence):
    return len(tokenizer.encode(sentence))

def get_model_data_think(model_name):
    model_path = os.path.join(label_dir, model_name)
    model_data_think = []
    
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} does not exist")
        return []

    for file in os.listdir(model_path):
        if not file.endswith('.json'):
            continue
            
        with open(os.path.join(model_path, file), "r") as f:
            data = json.load(f)
            
        new_data = []
        new_data_think = []
        
        for item in data:
            if item["sentence-type"] == "think":
                new_data_think.append({
                    "category": item["sentence-category"],
                    "length": get_token_length(item["sentence"])
                })
            
            # Logic from seq_l.py
            if len(new_data) == 0:
                new_data.append({
                    "category": item["sentence-category"],
                    "length": get_token_length(item["sentence"])
                })
            else:
                if item["sentence-category"] == new_data[-1]["category"]:
                    new_data[-1]["length"] += get_token_length(item["sentence"])
                    if item["sentence-type"] == "think":
                        new_data_think[-1]["length"] += get_token_length(item["sentence"])
                    continue
                new_data.append({
                    "category": item["sentence-category"],
                    "length": get_token_length(item["sentence"])
                })
                
        model_data_think.append(new_data_think)
    return model_data_think

def get_time_curve(data, num_bins):

    whole_category_bin = {
            'Read': [0] * num_bins,
            'Monitor': [0] * num_bins,
            'Plan': [0] * num_bins,
            'Implement': [0] * num_bins,
            'Analyze': [0] * num_bins,
            'Explore': [0] * num_bins,
            'Verify': [0] * num_bins,
            'Answer': [0] * num_bins,
    }

    for single_response in data:
        total_length = 0
        category_total_length = {
            'Read': 0,
            'Monitor': 0,
            'Plan': 0,
            'Implement': 0,
            'Analyze': 0,
            'Explore': 0,
            'Verify': 0,
            'Answer': 0
        }
        for item in single_response:
            total_length += item['length']
            category_total_length[item['category']] += item['length']
        
        bin_size = total_length / num_bins

        category_bin = {
            'Read': [0] * num_bins,
            'Monitor': [0] * num_bins,
            'Plan': [0] * num_bins,
            'Implement': [0] * num_bins,
            'Analyze': [0] * num_bins,
            'Explore': [0] * num_bins,
            'Verify': [0] * num_bins,
            'Answer': [0] * num_bins,
        }

        current_token = 0
        for item in single_response:
            category = item['category']
            length = item['length']
            
            # Process each token in the current item
            for i in range(length):
                # Determine which bin this token belongs to
                bin_idx = int(current_token / bin_size)
                
                # Make sure we don't exceed the number of bins (edge case for last bin)
                if bin_idx >= num_bins:
                    bin_idx = num_bins - 1
                
                # Increment the count for this category in this bin
                category_bin[category][bin_idx] += 1
                
                current_token += 1

        # normalize the category_bin
        for category in category_bin:
            category_bin[category] = [x / category_total_length[category] if category_total_length[category] > 0 else 1 / num_bins for x in category_bin[category]]
            for i in range(num_bins):
                whole_category_bin[category][i] += category_bin[category][i]

    # normalize the whole_category_bin by len(data)
    for category in whole_category_bin:
        whole_category_bin[category] = [x / len(data) for x in whole_category_bin[category]]
    
    return whole_category_bin

for model in models:
    # with open(os.path.join(base_path, model + '_think.json'), 'r') as f:
    #     data = json.load(f)
    print(f"Processing {model}...")
    data = get_model_data_think(model)
    time_curve = get_time_curve(data, 25)
    curve_data[model] = time_curve

model_name_map = {
    'deepseekR1': 'DeepSeek-R1',
    'Phi4R': 'Phi-4-Reasoning',
    'Qwen3_32B': 'Qwen-3-32B', 
    'QwQ32B': 'QwQ-32B',
    'DSQwen32B': 'R1-Distill-Qwen-32B'
}

# Get all categories
categories = ["Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Answer", "Monitor"]

# Create a figure with subplots for each category
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for idx, category in enumerate(categories):
    ax = axes[idx]
    
    # Collect data for all models for this category
    model_curves = []
    for model in models:
        curve = curve_data[model][category]
        # Apply smoothing
        smoothed_curve = gaussian_filter1d(curve, sigma=1.0)
        model_curves.append(smoothed_curve)
    
    # Convert to numpy array for easier computation
    model_curves = np.array(model_curves)
    
    # Calculate mean and standard error
    mean_curve = np.mean(model_curves, axis=0)
    std_error = np.std(model_curves, axis=0) / np.sqrt(len(models))
    
    # Plot mean curve over 0–100% of the response (percentage progress)
    x = np.linspace(0, 100, len(mean_curve))
    ax.plot(x, mean_curve, 'b-', linewidth=2, label='Mean' if idx == 0 else '')
    
    # Plot shaded region for standard error
    ax.fill_between(x, mean_curve - std_error, mean_curve + std_error, 
                     alpha=0.3, color='blue', label='±SD' if idx == 0 else '')
    
    # Plot individual model curves with transparency
    for i, model in enumerate(models):
        ax.plot(x, model_curves[i], '--', alpha=0.4, linewidth=1, 
                label=model if idx == 0 else '')
    
    ax.set_title(category, fontsize=12, fontweight='bold')
    ax.set_xlabel('Response Progress (%)', fontsize=12, fontweight='bold')
    
    # Only show legend on the first subplot (Read)
    # if idx == 0:
    #     ax.legend(loc='best', fontsize=8)
    if idx == 0 or idx == 4:
        ax.set_ylabel('Normalized Frequency', fontsize=12, fontweight='bold')
    
    # Bold tick labels
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3)

# Add a single legend for the entire figure at the bottom
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [model_name_map[label] if label in model_name_map else label for label in labels]
fig.legend(handles, new_labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=7, prop={'weight': 'bold', 'size': 10})

fig.align_ylabels([axes[0], axes[4]])

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('analysis/temporal_plot.pdf', dpi=300)
plt.close()