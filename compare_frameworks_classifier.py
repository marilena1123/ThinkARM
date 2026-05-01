"""
Compare ThinkARM and BLOOM for correctness prediction.

Trains Lasso logistic regression classifiers using:
1. ThinkARM episode features
2. BLOOM cognitive level features

Compares:
- Model performance (accuracy, AUC, precision, recall)
- Feature importance
- Which framework better predicts correctness
- Feature overlap and differences

Usage:
    python compare_frameworks_classifier.py \
        --thinkarm_label_dir data/label \
        --bloom_annotated_dir outputs_bloom_annotated \
        --correctness_dir data/correct \
        --models deepseekR1,DeepSeek-R1-Distill-Qwen-7B,Phi4R
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

THINKARM_CATEGORIES = [
    "Read", "Analyze", "Plan", "Implement",
    "Explore", "Verify", "Monitor", "Answer"
]

BLOOM_CATEGORIES = [
    "REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE", "EVALUATE", "CREATE"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ThinkARM vs BLOOM for correctness prediction"
    )
    parser.add_argument(
        "--thinkarm_label_dir",
        type=str,
        default="data/label",
        help="Path to ThinkARM label directory",
    )
    parser.add_argument(
        "--bloom_annotated_dir",
        type=str,
        default="outputs_bloom_annotated",
        help="Path to BLOOM annotated results",
    )
    parser.add_argument(
        "--correctness_dir",
        type=str,
        default="data/correct",
        help="Path to correctness labels",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated models to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory to save comparison results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per model",
    )
    return parser.parse_args()


def estimate_tokens(text):
    """Estimate token count using GPT-4 tokenizer."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        return len(text.split())  # Fallback to word count


# ===== ThinkARM Feature Extraction =====

def extract_thinkarm_features(sentences):
    """Extract features from ThinkARM sentence-level annotations."""
    features = {}
    total_tokens = 0
    think_tokens = 0
    answer_tokens = 0
    cat_token_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    prev_cat = None

    for item in sentences:
        text = item.get("sentence", "")
        cat = item.get("sentence-category", "Monitor")
        stype = item.get("sentence-type", "think")
        n_tokens = estimate_tokens(text)

        total_tokens += n_tokens
        if stype == "answer":
            answer_tokens += n_tokens
        else:
            think_tokens += n_tokens

        if cat in THINKARM_CATEGORIES:
            cat_token_counts[cat] += n_tokens

        if prev_cat is not None:
            transition_counts[(prev_cat, cat)] += 1
        prev_cat = cat

    safe_total = total_tokens if total_tokens > 0 else 1

    # Global statistics
    features["TA_Total_Tokens"] = total_tokens
    features["TA_Think_Ratio"] = think_tokens / safe_total

    # Episode intensity
    for cat in THINKARM_CATEGORIES:
        cnt = cat_token_counts[cat]
        features[f"TA_Ratio_{cat}"] = cnt / safe_total

    # Transitions
    for src in THINKARM_CATEGORIES:
        for tgt in THINKARM_CATEGORIES:
            count = transition_counts[(src, tgt)]
            features[f"TA_Trans_{src}_to_{tgt}"] = count

    return features


# ===== BLOOM Feature Extraction =====

def extract_bloom_features(labels):
    """Extract features from BLOOM cognitive level annotations."""
    features = {}
    total_tokens = 0
    cat_token_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    prev_cat = None

    for item in labels:
        step_text = item.get("step", "")
        cat = item.get("bloom_level", "UNDERSTAND")
        n_tokens = estimate_tokens(step_text)

        total_tokens += n_tokens

        if cat in BLOOM_CATEGORIES:
            cat_token_counts[cat] += n_tokens

        if prev_cat is not None:
            transition_counts[(prev_cat, cat)] += 1
        prev_cat = cat

    safe_total = total_tokens if total_tokens > 0 else 1

    # Global statistics
    features["BL_Total_Tokens"] = total_tokens

    # Cognitive level intensity
    for cat in BLOOM_CATEGORIES:
        cnt = cat_token_counts[cat]
        features[f"BL_Ratio_{cat}"] = cnt / safe_total

    # Transitions
    for src in BLOOM_CATEGORIES:
        for tgt in BLOOM_CATEGORIES:
            count = transition_counts[(src, tgt)]
            features[f"BL_Trans_{src}_to_{tgt}"] = count

    return features


# ===== Data Loading =====

def load_data(args):
    """Load data from both frameworks."""
    data_points = []

    models = args.models.split(",") if args.models else None

    for model_name in os.listdir(args.thinkarm_label_dir):
        if models and model_name not in models:
            continue

        label_dir = Path(args.thinkarm_label_dir) / model_name
        if not label_dir.exists():
            continue

        # Load correctness labels
        correctness_file = Path(args.correctness_dir) / f"{model_name}.json"
        if not correctness_file.exists():
            print(f"  Skipping {model_name}: no correctness file")
            continue

        try:
            with open(correctness_file, "r") as f:
                correctness_map = json.load(f)
        except:
            continue

        # Load BLOOM annotations for this model
        bloom_files = list(Path(args.bloom_annotated_dir).glob(f"*{model_name}*.json"))
        bloom_data = {}
        for bloom_file in bloom_files:
            try:
                with open(bloom_file, "r") as f:
                    data = json.load(f)
                    for result in data.get("results", []):
                        problem_id = result.get("problem_id", -1)
                        bloom_data[problem_id] = result
            except:
                pass

        print(f"\nProcessing {model_name}...")
        count = 0

        # Process each problem
        for json_file in sorted(label_dir.glob("*.json")):
            if args.max_samples and count >= args.max_samples:
                break

            problem_id = int(json_file.stem)

            # Get correctness
            is_correct = correctness_map.get(str(problem_id)) or correctness_map.get(problem_id)
            if is_correct is None:
                continue

            # Load ThinkARM annotations
            try:
                with open(json_file, "r") as f:
                    ta_sentences = json.load(f)

                # Check for both think and answer
                types = {item.get("sentence-type") for item in ta_sentences}
                if "think" not in types or "answer" not in types:
                    continue

                ta_features = extract_thinkarm_features(ta_sentences)

            except:
                continue

            # Load BLOOM annotations
            bl_features = {}
            if problem_id in bloom_data:
                bloom_result = bloom_data[problem_id]
                bl_labels = bloom_result.get("bloom_labels", [])
                if bl_labels:
                    bl_features = extract_bloom_features(bl_labels)

            # Combine features
            all_features = {**ta_features, **bl_features}
            all_features["correctness"] = 1 if is_correct else 0
            all_features["model"] = model_name
            all_features["problem_id"] = problem_id

            data_points.append(all_features)
            count += 1

        print(f"  Loaded {count} problems for {model_name}")

    return pd.DataFrame(data_points) if data_points else None


# ===== Classification & Comparison =====

def train_and_evaluate(X, y, X_test, y_test, feature_names, framework_name):
    """Train classifier and return results."""
    print(f"\n  Training {framework_name}...")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # Train
    clf = LogisticRegression(
        penalty="l1",
        C=0.5,
        solver="liblinear",
        max_iter=2000,
        random_state=42
    )
    clf.fit(X_scaled, y)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    results = {
        "framework": framework_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "coefficients": pd.DataFrame({
            "feature": feature_names,
            "coefficient": clf.coef_[0],
            "abs_coef": np.abs(clf.coef_[0])
        })
    }

    return results


def compare_frameworks(df, output_dir):
    """Compare ThinkARM and BLOOM frameworks."""
    os.makedirs(output_dir, exist_ok=True)

    # Get feature columns
    ta_features = [c for c in df.columns if c.startswith("TA_")]
    bl_features = [c for c in df.columns if c.startswith("BL_")]
    drop_cols = ["correctness", "model", "problem_id"] + [c for c in df.columns if c not in ta_features + bl_features + ["correctness"]]

    print(f"\nFramework Comparison:")
    print(f"  ThinkARM features: {len(ta_features)}")
    print(f"  BLOOM features: {len(bl_features)}")
    print(f"  Total samples: {len(df)}")

    # Split data (80/20)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=df["correctness"]
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    results = {}

    # ThinkARM
    X_train = train_df[ta_features].fillna(0).values
    X_test = test_df[ta_features].fillna(0).values
    y_train = train_df["correctness"].values
    y_test = test_df["correctness"].values

    results["ThinkARM"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, np.array(ta_features), "ThinkARM"
    )

    # BLOOM
    X_train = train_df[bl_features].fillna(0).values
    X_test = test_df[bl_features].fillna(0).values

    results["BLOOM"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, np.array(bl_features), "BLOOM"
    )

    # Combined
    all_features = ta_features + bl_features
    X_train = train_df[all_features].fillna(0).values
    X_test = test_df[all_features].fillna(0).values

    results["Combined"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, np.array(all_features), "Combined"
    )

    return results


def save_comparison_report(results, output_dir):
    """Save detailed comparison report."""
    report_file = Path(output_dir) / "framework_comparison_report.txt"

    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FRAMEWORK COMPARISON: ThinkARM vs BLOOM for Correctness Prediction\n")
        f.write("=" * 70 + "\n\n")

        # Performance comparison
        f.write("1. PERFORMANCE COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Framework':<15} {'Accuracy':<12} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 70 + "\n")

        for framework, res in results.items():
            f.write(
                f"{framework:<15} {res['accuracy']:.4f}       {res['auc']:.4f}       "
                f"{res['precision']:.4f}       {res['recall']:.4f}       {res['f1']:.4f}\n"
            )

        # Top features
        f.write("\n\n2. TOP PREDICTIVE FEATURES\n")
        f.write("-" * 70 + "\n")

        for framework, res in results.items():
            f.write(f"\n### {framework} ###\n")
            coefs = res["coefficients"].copy()

            # Positive features
            pos = coefs[coefs["coefficient"] > 0].sort_values("abs_coef", ascending=False).head(10)
            f.write(f"\n  Positive Contributors (Help Correctness):\n")
            for _, row in pos.iterrows():
                f.write(f"    {row['feature']:<40} {row['coefficient']:>8.4f}\n")

            # Negative features
            neg = coefs[coefs["coefficient"] < 0].sort_values("abs_coef", ascending=False).head(10)
            f.write(f"\n  Negative Contributors (Hurt Correctness):\n")
            for _, row in neg.iterrows():
                f.write(f"    {row['feature']:<40} {row['coefficient']:>8.4f}\n")

    print(f"\n✅ Report saved: {report_file}")


def main():
    args = parse_args()

    print("=" * 70)
    print("FRAMEWORK COMPARISON: ThinkARM vs BLOOM")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data(args)

    if df is None or len(df) == 0:
        print("ERROR: No data loaded!")
        return

    print(f"\nLoaded {len(df)} total samples")

    # Compare frameworks
    results = compare_frameworks(df, args.output_dir)

    # Save report
    save_comparison_report(results, args.output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for framework, res in results.items():
        print(f"\n{framework}:")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  AUC:      {res['auc']:.4f}")
        print(f"  F1:       {res['f1']:.4f}")


if __name__ == "__main__":
    main()
