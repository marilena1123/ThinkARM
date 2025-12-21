import json
import os

label_dir = "data/label"
seq_dir = "data/episode_ngram"
os.makedirs(seq_dir, exist_ok=True)
os.makedirs(os.path.join(seq_dir, "model"), exist_ok=True)
models = os.listdir(label_dir)
# annotator = "gpt-5"

label_map = {
    "Read": 'R',
    "Monitor": 'M',
    "Plan": 'P',
    "Analyze": 'N',
    "Explore": 'E',
    "Implement": 'I',
    "Answer": 'A',
    "Verify": 'V',
}

for model in models:
    model_data = []
    model_data_think = []
    model_data_answer = []
    for file in os.listdir(os.path.join(label_dir, model)):
        with open(os.path.join(label_dir, model, file), "r") as f:
            data = json.load(f)
        new_data = ''
        new_data_think = ''
        new_data_answer = ''
        for item in data:
            if new_data == '':
                new_data += label_map[item["sentence-category"]]
            else:
                if label_map[item["sentence-category"]] == new_data[-1]:
                    continue
                new_data += label_map[item["sentence-category"]]
            if item["sentence-type"] == "think":
                new_data_think += label_map[item["sentence-category"]]
            if item["sentence-type"] == "answer":
                new_data_answer += label_map[item["sentence-category"]]
        model_data.append(new_data)
        model_data_think.append(new_data_think)
        model_data_answer.append(new_data_answer)
    with open(os.path.join(seq_dir, f"model/{model}.json"), "w") as f:
        json.dump(model_data, f, indent=4)
    with open(os.path.join(seq_dir, f"model/{model}_think.json"), "w") as f:
        json.dump(model_data_think, f, indent=4)
    with open(os.path.join(seq_dir, f"model/{model}_answer.json"), "w") as f:
        json.dump(model_data_answer, f, indent=4)


reasoning_models = ["deepseekR1", "Qwen3_32B", "QwQ32B", "Phi4R", "DSQwen32B", "o1mini", "o3mini","gemini2.5flash"]
non_reasoning_models = ["gpt4o", "Phi4", "Qwen3_32BNR", "gemini2.0flash"]

reasoning_models_distill = [
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Qwen-1.5B"
    ]


efficient_models_alpha = [
    "alpha_0.4_DeepSeek-R1-Distill-Qwen-1.5B",
    # "alpha_0.4_DeepSeek-R1-Distill-Qwen-7B",
]
efficient_models_thinkprune = [
    # "DeepScaleR-1.5B-Preview-thinkprune-iter2k",
    "DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k",
]
efficient_models_answer_l1 = [
      "L1-Qwen-1.5B-Max",
    #   "L1-Qwen-7B-Max",
]
reasoning_models_distill = [
    "DeepSeek-R1-Distill-Qwen-1.5B",
    # "DeepSeek-R1-Distill-Qwen-7B",
]

reasoning_seq = []
non_reasoning_seq = []
reasoning_seq_think = []
reasoning_seq_answer = []
efficient_seq_alpha = []
efficient_seq_alpha_think = []
efficient_seq_alpha_answer = []
efficient_seq_thinkprune = []
efficient_seq_thinkprune_think = []
efficient_seq_thinkprune_answer = []
efficient_seq_answer_l1 = []
efficient_seq_answer_l1_think = []
efficient_seq_answer_l1_answer = []
reasoning_seq_distill = []
reasoning_seq_distill_think = []
reasoning_seq_distill_answer = []
all_seq = []

excluded_models = ["o1mini", "o3mini", "gemini2.5flash"]

for model in reasoning_models:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_think.json"), "r") as f:
        data_think = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_answer.json"), "r") as f:
        data_answer = json.load(f)
        
    for i in range(len(data)):
        if model not in excluded_models:
            # Check if any part is empty
            if data[i] == "" or data_think[i] == "" or data_answer[i] == "":
                print(f"Skipping {model} index {i} due to empty value")
                continue
        
        reasoning_seq.append(data[i])
        if model in excluded_models:
             reasoning_seq_think.append("") 
        else:
             reasoning_seq_think.append(data_think[i])
             
        reasoning_seq_answer.append(data_answer[i])
        all_seq.append(data[i])

# Process efficient_models_alpha
for model in efficient_models_alpha:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_think.json"), "r") as f:
        data_think = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_answer.json"), "r") as f:
        data_answer = json.load(f)
        
    for i in range(len(data)):
        if model not in excluded_models:
            # Check if any part is empty
            if data[i] == "" or data_think[i] == "" or data_answer[i] == "":
                print(f"Skipping {model} index {i} due to empty value")
                continue
        
        efficient_seq_alpha.append(data[i])
        efficient_seq_alpha_think.append(data_think[i])
        efficient_seq_alpha_answer.append(data_answer[i])
        all_seq.append(data[i])

# Process efficient_models_thinkprune
for model in efficient_models_thinkprune:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_think.json"), "r") as f:
        data_think = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_answer.json"), "r") as f:
        data_answer = json.load(f)
        
    for i in range(len(data)):
        if model not in excluded_models:
            # Check if any part is empty
            if data[i] == "" or data_think[i] == "" or data_answer[i] == "":
                print(f"Skipping {model} index {i} due to empty value")
                continue
        
        efficient_seq_thinkprune.append(data[i])
        efficient_seq_thinkprune_think.append(data_think[i])
        efficient_seq_thinkprune_answer.append(data_answer[i])
        all_seq.append(data[i])

# Process efficient_models_answer_l1
for model in efficient_models_answer_l1:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_think.json"), "r") as f:
        data_think = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_answer.json"), "r") as f:
        data_answer = json.load(f)
        
    for i in range(len(data)):
        if model not in excluded_models:
            # Check if any part is empty
            if data[i] == "" or data_think[i] == "" or data_answer[i] == "":
                print(f"Skipping {model} index {i} due to empty value")
                continue
        
        efficient_seq_answer_l1.append(data[i])
        efficient_seq_answer_l1_think.append(data_think[i])
        efficient_seq_answer_l1_answer.append(data_answer[i])
        all_seq.append(data[i])

# Process reasoning_models_distill
for model in reasoning_models_distill:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_think.json"), "r") as f:
        data_think = json.load(f)
    with open(os.path.join(seq_dir, f"model/{model}_answer.json"), "r") as f:
        data_answer = json.load(f)
        
    for i in range(len(data)):
        if model not in excluded_models:
            # Check if any part is empty
            if data[i] == "" or data_think[i] == "" or data_answer[i] == "":
                print(f"Skipping {model} index {i} due to empty value")
                continue
        
        reasoning_seq_distill.append(data[i])
        reasoning_seq_distill_think.append(data_think[i])
        reasoning_seq_distill_answer.append(data_answer[i])
        all_seq.append(data[i])

for model in non_reasoning_models:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        data = json.load(f)
        non_reasoning_seq.extend(data)
        all_seq.extend(data)

with open(os.path.join(seq_dir, "reasoning.json"), "w") as f:
    json.dump(reasoning_seq, f, indent=4)
with open(os.path.join(seq_dir, "reasoning_think.json"), "w") as f:
    json.dump(reasoning_seq_think, f, indent=4)
with open(os.path.join(seq_dir, "reasoning_answer.json"), "w") as f:
    json.dump(reasoning_seq_answer, f, indent=4)
with open(os.path.join(seq_dir, "non_reasoning.json"), "w") as f:
    json.dump(non_reasoning_seq, f, indent=4)

# efficient_seq_alpha
with open(os.path.join(seq_dir, "efficient_alpha.json"), "w") as f:
    json.dump(efficient_seq_alpha, f, indent=4)
with open(os.path.join(seq_dir, "efficient_alpha_think.json"), "w") as f:
    json.dump(efficient_seq_alpha_think, f, indent=4)
with open(os.path.join(seq_dir, "efficient_alpha_answer.json"), "w") as f:
    json.dump(efficient_seq_alpha_answer, f, indent=4)

# efficient_seq_thinkprune
with open(os.path.join(seq_dir, "efficient_thinkprune.json"), "w") as f:
    json.dump(efficient_seq_thinkprune, f, indent=4)
with open(os.path.join(seq_dir, "efficient_thinkprune_think.json"), "w") as f:
    json.dump(efficient_seq_thinkprune_think, f, indent=4)
with open(os.path.join(seq_dir, "efficient_thinkprune_answer.json"), "w") as f:
    json.dump(efficient_seq_thinkprune_answer, f, indent=4)

# efficient_seq_answer_l1
with open(os.path.join(seq_dir, "efficient_l1.json"), "w") as f:
    json.dump(efficient_seq_answer_l1, f, indent=4)
with open(os.path.join(seq_dir, "efficient_l1_think.json"), "w") as f:
    json.dump(efficient_seq_answer_l1_think, f, indent=4)
with open(os.path.join(seq_dir, "efficient_l1_answer.json"), "w") as f:
    json.dump(efficient_seq_answer_l1_answer, f, indent=4)

# reasoning_seq_distill
with open(os.path.join(seq_dir, "reasoning_distill.json"), "w") as f:
    json.dump(reasoning_seq_distill, f, indent=4)
with open(os.path.join(seq_dir, "reasoning_distill_think.json"), "w") as f:
    json.dump(reasoning_seq_distill_think, f, indent=4)
with open(os.path.join(seq_dir, "reasoning_distill_answer.json"), "w") as f:
    json.dump(reasoning_seq_distill_answer, f, indent=4)

with open(os.path.join(seq_dir, "all.json"), "w") as f:
    json.dump(all_seq, f, indent=4)