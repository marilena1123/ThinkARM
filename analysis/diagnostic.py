import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================
# Part 1: Feature Extraction Logic 
# ==========================================

CATEGORIES = [
    "Read", "Analyze", "Plan", "Implement", 
    "Explore", "Verify", "Monitor", "Answer"
]

def estimate_tokens(text):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base") # GPT-4 tokenizer
    return len(enc.encode(text))

def extract_features(data_list):
    features = {}
    total_tokens = 0
    think_tokens = 0
    answer_tokens = 0
    cat_token_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    prev_cat = None
    
    for item in data_list:
        text = item.get("sentence", "")
        cat = item.get("sentence-category", "Monitor") 
        sType = item.get("sentence-type", "think")
        n_tokens = estimate_tokens(text)
        
        total_tokens += n_tokens
        if sType == "answer": answer_tokens += n_tokens
        else: think_tokens += n_tokens
            
        if cat in CATEGORIES: cat_token_counts[cat] += n_tokens
        if prev_cat is not None: transition_counts[(prev_cat, cat)] += 1
        prev_cat = cat

    safe_total = total_tokens if total_tokens > 0 else 1
    features["Total_Tokens"] = total_tokens
    features["Think_Ratio"] = think_tokens / safe_total
    for cat in CATEGORIES:
        cnt = cat_token_counts[cat]
        # features[f"Token_Count_{cat}"] = cnt
        features[f"Token_Ratio_{cat}"] = cnt / safe_total
    for src in CATEGORIES:
        for tgt in CATEGORIES:
            count = transition_counts[(src, tgt)]
            features[f"Trans_{src}_to_{tgt}"] = count
    return features

# ==========================================
# Part 2: Full Dataset Analysis Logic
# ==========================================

def save_full_report(results_df, filename="analysis/cognitive_features_full_dataset_report.txt"):
    """
    Saves the feature report based on the full dataset model.
    """
    # Filter out features that were zeroed out by Lasso
    active_df = results_df[results_df['Coefficient'] != 0].copy()
    
    # Split into Positive and Negative
    pos_df = active_df[active_df['Coefficient'] > 0].sort_values(by='Coefficient', ascending=False)
    neg_df = active_df[active_df['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=========================================================\n")
        f.write("      FULL DATASET COGNITIVE FEATURE REPORT (Overfit)    \n")
        f.write("=========================================================\n\n")
        
        # --- Positive Section ---
        f.write(f"=== POSITIVE CONTRIBUTORS (Help Correctness: {len(pos_df)} features) ===\n")
        f.write(f"{'Feature Name':<35} | {'Coefficient':<12}\n")
        f.write("-" * 50 + "\n")
        for _, row in pos_df.iterrows():
            f.write(f"{row['Feature']:<35} | {row['Coefficient']:<12.4f}\n")
        
        f.write("\n\n")
        
        # --- Negative Section ---
        f.write(f"=== NEGATIVE CONTRIBUTORS (Hurt Correctness: {len(neg_df)} features) ===\n")
        f.write(f"{'Feature Name':<35} | {'Coefficient':<12}\n")
        f.write("-" * 50 + "\n")
        for _, row in neg_df.iterrows():
            f.write(f"{row['Feature']:<35} | {row['Coefficient']:<12.4f}\n")

    print(f"\n[Saved] Full dataset feature report saved to: {filename}")

def run_full_dataset_analysis(df):
    """
    Runs Lasso Logistic Regression on the ENTIRE dataset without cross-validation.
    """
    # 1. Prepare Data
    drop_cols = ['correctness', 'model', 'q_id']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].fillna(0).values
    y = df['correctness'].values
    feature_names = np.array(feature_cols)
    
    print(f"\nTraining on FULL dataset ({len(X)} samples)...")
    
    # 2. Scaling (Important for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train Lasso (L1) on ALL data
    # Note: You can adjust C. Larger C = less regularization (more features, more overfitting). 
    # C=0.5 is good for feature selection. C=1.0 might improve Train Accuracy.
    clf = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', max_iter=2000, random_state=42)
    clf.fit(X_scaled, y)
    
    # 4. Evaluate on Training Data (Self-Evaluation / Overfitting check)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    
    train_acc = accuracy_score(y, y_pred)
    train_auc = roc_auc_score(y, y_prob)
    
    # 5. Extract Coefficients
    coefs = clf.coef_[0]
    
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coef': np.abs(coefs)
    })
    
    # Save Report
    save_full_report(results_df)
    
    # Print Performance
    print("\n" + "="*40)
    print("FULL DATASET MODEL PERFORMANCE (Training Set)")
    print("="*40)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training ROC AUC:  {train_auc:.4f}")
    print("(Note: These metrics indicate how well the features DESCRIBE the current data,")
    print(" not necessarily how well they generalize to unseen data.)")
    print("="*40)
    
    return results_df

# ==========================================
# Part 3: Main Execution Flow
# ==========================================

def main():
    models = ["deepseekR1", "DSQwen32B", "Phi4R", "Qwen3_32B", "QwQ32B"]
    data_points = []
    
    print("--- Phase 1: Loading Data ---")
    skipped_count = 0
    for model in models:
        label_dir = os.path.join("data", "label", model)
        correctness_file = os.path.join("data", "correct", f"{model}.json")
        
        if not os.path.exists(correctness_file) or not os.path.exists(label_dir): continue
        with open(correctness_file, 'r', encoding='utf-8') as f: correctness_map = json.load(f)
        
        for json_file in os.listdir(label_dir):
            if not json_file.endswith('.json'): continue
            q_id = os.path.splitext(json_file)[0]
            
            if q_id not in correctness_map:
                try: 
                    if int(q_id) in correctness_map: pass
                    else: continue
                except: continue
                    
            is_correct = correctness_map[q_id]
            
            try:
                with open(os.path.join(label_dir, json_file), 'r', encoding='utf-8') as f: content = json.load(f)
                
                types = {item.get("sentence-type") for item in content}
                if "think" not in types or "answer" not in types:
                    skipped_count += 1
                    continue
                    
                feats = extract_features(content)
                feats['correctness'] = 1 if is_correct else 0
                feats['model'] = model
                feats['q_id'] = q_id
                data_points.append(feats)
            except: continue

    if not data_points: 
        print("No data found.")
        return

    df = pd.DataFrame(data_points)
    print(f"Loaded {len(df)} samples.")
    
    # --- Phase 2: Run Full Dataset Analysis ---
    results_df = run_full_dataset_analysis(df)
    
    # Display Top Features
    print("\nTop Positive Contributors (Help Correctness):")
    print(results_df[results_df['Coefficient'] > 0]
          .sort_values(by='Coefficient', ascending=False)
          .head(10)[['Feature', 'Coefficient']]
          .to_string(index=False))
          
    print("\nTop Negative Contributors (Hurt Correctness):")
    print(results_df[results_df['Coefficient'] < 0]
          .sort_values(by='Coefficient', ascending=True)
          .head(10)[['Feature', 'Coefficient']]
          .to_string(index=False))

if __name__ == "__main__":
    main()