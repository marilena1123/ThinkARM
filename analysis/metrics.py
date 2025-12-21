import json
import os
from sklearn.metrics import cohen_kappa_score
import numpy as np

response_models = ["deepseekR1", "Phi4R", "Qwen3_32B", "QwQ32B", "gpt4o", "Qwen3_32BNR", "Phi4", "gemini2.0flash"]
annotators = ["gpt-5", "gpt-4.1", "gemini-2.5-pro", "gemini-2.5-flash"]
label_path = "data/label"
gt_path = "data/ground_truth"
for annotator in annotators:
    gt_labels = []
    label_labels = []
    for response_model in response_models:
        files = [str(i).json for i in range(1, 10)]
        for file in files:
            with open(os.path.join(label_path, response_model, annotator, file), 'r') as f:
                label_data = json.load(f)
            with open(os.path.join(gt_path, response_model, file), 'r') as f:
                gt_data = json.load(f)
            gt_labels.extend([item['human_label'] for item in gt_data])
            label_labels.extend([item['sentence-category'] for item in label_data])
    
    print("="*100)
    print(f"Annotator: {annotator}")
    print(f"Accuracy: {np.mean(np.array(gt_labels) == np.array(label_labels))}")
    print(f"Kappa: {cohen_kappa_score(gt_labels, label_labels)}")       