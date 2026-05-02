"""
Compare ThinkARM and BLOOM using THINKING TRACES ONLY.

Analyzes only the <think> blocks to predict correctness based on
the reasoning process itself, excluding final answers.

Feature specifications:
- ThinkARM: 75 features (8 episode ratios + 64 transitions + 2 dynamics)
- BLOOM: 44 features (1 total tokens + 1 step count + 6 ratios + 36 normalized transitions)

Trains Lasso logistic regression classifiers using:
1. ThinkARM episodes (thinking only)
2. BLOOM cognitive levels (thinking only) - published spec
3. Combined features

Usage:
    python compare_frameworks_thinking_only.py \
        --thinkarm_label_dir data/label \
        --bloom_annotated_dir outputs_bloom_annotated \
        --correctness_dir data/correct \
        --models deepseekR1,Phi4R,DeepSeek-R1-Distill-Qwen-7B \
        --output_dir comparison_thinking_results
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
from sklearn.model_selection import train_test_split

THINKARM_CATEGORIES = [
    "Read", "Analyze", "Plan", "Implement",
    "Explore", "Verify", "Monitor", "Answer"
]

BLOOM_CATEGORIES = [
    "REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE", "EVALUATE", "CREATE"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ThinkARM vs BLOOM on thinking traces only"
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
        default="comparison_thinking_results",
        help="Directory to save results",
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
        return len(text.split())


# ===== ThinkARM Feature Extraction (Thinking Only) =====

def extract_thinkarm_thinking_features(sentences):
    """Extract features from ThinkARM THINKING TRACES ONLY."""
    features = {}

    # Filter only "think" sentences
    think_sentences = [s for s in sentences if s.get("sentence-type") == "think"]

    if not think_sentences:
        return None

    total_tokens = 0
    cat_token_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    prev_cat = None

    for item in think_sentences:
        text = item.get("sentence", "")
        cat = item.get("sentence-category", "Monitor")
        n_tokens = estimate_tokens(text)

        total_tokens += n_tokens

        if cat in THINKARM_CATEGORIES:
            cat_token_counts[cat] += n_tokens

        if prev_cat is not None:
            transition_counts[(prev_cat, cat)] += 1
        prev_cat = cat

    safe_total = total_tokens if total_tokens > 0 else 1

    # Global statistics (thinking only)
    features["TA_Think_Total_Tokens"] = total_tokens

    # Episode intensity in thinking
    for cat in THINKARM_CATEGORIES:
        cnt = cat_token_counts[cat]
        features[f"TA_Think_Ratio_{cat}"] = cnt / safe_total

    # Transitions within thinking
    for src in THINKARM_CATEGORIES:
        for tgt in THINKARM_CATEGORIES:
            count = transition_counts[(src, tgt)]
            features[f"TA_Think_Trans_{src}_to_{tgt}"] = count

    # Compute episode dynamics
    # How many times does the model return to Explore or Monitor?
    explore_count = sum(1 for s in think_sentences if s.get("sentence-category") == "Explore")
    monitor_count = sum(1 for s in think_sentences if s.get("sentence-category") == "Monitor")

    features["TA_Think_Explore_Freq"] = explore_count / max(1, len(think_sentences))
    features["TA_Think_Monitor_Freq"] = monitor_count / max(1, len(think_sentences))

    return features


# ===== BLOOM Feature Extraction (Thinking Only) =====

def extract_bloom_thinking_features(labels):
    """Extract features from BLOOM THINKING TRACES ONLY.

    44 features (published spec):
    - Total tokens (1)
    - Step count (1)
    - Token proportions per BLOOM level (6)
    - Normalized pairwise transitions between adjacent steps (36)
    """
    features = {}

    if not labels:
        return None

    total_tokens = 0
    step_count = len(labels)
    cat_token_counts = defaultdict(int)
    transition_counts = defaultdict(int)

    for item in labels:
        step_text = item.get("step", "")
        cat = item.get("bloom_level", "UNDERSTAND")
        n_tokens = estimate_tokens(step_text)

        total_tokens += n_tokens

        if cat in BLOOM_CATEGORIES:
            cat_token_counts[cat] += n_tokens

    safe_total = total_tokens if total_tokens > 0 else 1

    # Feature 1: Total tokens
    features["BL_Think_Total_Tokens"] = total_tokens

    # Feature 2: Step count
    features["BL_Think_Step_Count"] = step_count

    # Features 3-8: Token proportions per cognitive level
    for cat in BLOOM_CATEGORIES:
        cnt = cat_token_counts[cat]
        features[f"BL_Think_Ratio_{cat}"] = cnt / safe_total

    # Features 9-44: Normalized pairwise transitions between adjacent steps
    # Count transitions only between consecutive steps
    total_transitions = 0
    for i in range(len(labels) - 1):
        src = labels[i].get("bloom_level", "UNDERSTAND")
        tgt = labels[i + 1].get("bloom_level", "UNDERSTAND")
        if src in BLOOM_CATEGORIES and tgt in BLOOM_CATEGORIES:
            transition_counts[(src, tgt)] += 1
            total_transitions += 1

    # Normalize transitions by total number of transitions
    norm_factor = total_transitions if total_transitions > 0 else 1

    for src in BLOOM_CATEGORIES:
        for tgt in BLOOM_CATEGORIES:
            count = transition_counts[(src, tgt)]
            features[f"BL_Think_Trans_{src}_to_{tgt}"] = count / norm_factor

    return features


# ===== Data Loading =====

def load_thinking_data(args):
    """Load data from both frameworks (thinking only)."""
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
        except Exception as e:
            print(f"  Error loading correctness: {e}")
            continue

        # Handle different correctness map formats
        # Convert all keys to integers for consistent lookup
        correctness_normalized = {}
        for key, value in correctness_map.items():
            try:
                idx = int(key) if isinstance(key, str) else key
                correctness_normalized[idx] = value
            except:
                pass

        # Load BLOOM thinking annotations
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

        print(f"\nProcessing {model_name} (thinking only)...")
        count = 0
        skipped = 0

        # Process each problem
        for json_file in sorted(label_dir.glob("*.json")):
            if args.max_samples and count >= args.max_samples:
                break

            problem_id = int(json_file.stem)

            # Get correctness - try multiple key formats
            is_correct = correctness_normalized.get(problem_id)
            if is_correct is None:
                skipped += 1
                continue

            # Load ThinkARM annotations
            try:
                with open(json_file, "r") as f:
                    ta_sentences = json.load(f)

                ta_features = extract_thinkarm_thinking_features(ta_sentences)
                if ta_features is None:
                    continue

            except:
                continue

            # Load BLOOM thinking annotations
            bl_features = {}
            if problem_id in bloom_data:
                bloom_result = bloom_data[problem_id]
                bl_labels = bloom_result.get("thinking_bloom_labels", [])
                if bl_labels:
                    bl_features = extract_bloom_thinking_features(bl_labels)

            if not bl_features:
                continue

            # Combine features
            all_features = {**ta_features, **bl_features}

            # Ensure correctness is 0 or 1
            correctness_label = 1 if is_correct else 0
            all_features["correctness"] = correctness_label
            all_features["model"] = model_name
            all_features["problem_id"] = problem_id

            data_points.append(all_features)
            count += 1

            # Debug output for first few samples
            if count <= 3:
                print(f"    Sample {problem_id}: is_correct={is_correct} -> label={correctness_label}")

        correct_count = sum(1 for d in data_points if d.get("correctness") == 1)
        incorrect_count = len(data_points) - correct_count if data_points else 0
        print(f"  Loaded {count} thinking traces for {model_name}")
        print(f"    Correct: {correct_count}, Incorrect: {incorrect_count} (skipped {skipped})")

    return pd.DataFrame(data_points) if data_points else None


# ===== Classification & Comparison =====

def train_and_evaluate(X, y, X_test, y_test, feature_names, framework_name):
    """Train classifier and return results."""
    print(f"    Training {framework_name}...")

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
        random_state=42,
        class_weight="balanced"  # Handle imbalanced classes
    )
    clf.fit(X_scaled, y)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    results = {
        "framework": framework_name,
        "n_features": len(feature_names),
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
    """Compare ThinkARM and BLOOM on thinking traces."""
    os.makedirs(output_dir, exist_ok=True)

    # Get feature columns
    ta_features = [c for c in df.columns if c.startswith("TA_")]
    bl_features = [c for c in df.columns if c.startswith("BL_")]

    print(f"\n{'='*70}")
    print("THINKING TRACES ONLY - Framework Comparison")
    print(f"{'='*70}")
    print(f"  ThinkARM features: {len(ta_features)}")
    print(f"  BLOOM features: {len(bl_features)}")
    print(f"  Total samples: {len(df)}")
    print(f"  Correct: {(df['correctness']==1).sum()} | Incorrect: {(df['correctness']==0).sum()}")

    # Split data - handle small datasets
    # Check if we can stratify (need at least 2 samples per class)
    unique_classes = df["correctness"].unique()
    can_stratify = all(
        (df["correctness"] == c).sum() >= 2
        for c in unique_classes
    )

    if can_stratify:
        train_idx, test_idx = train_test_split(
            range(len(df)), test_size=0.2, random_state=42, stratify=df["correctness"]
        )
    else:
        # Small dataset: use simple split without stratification
        test_size = max(1, int(len(df) * 0.2))
        test_idx = list(range(len(df) - test_size, len(df)))
        train_idx = list(range(len(df) - test_size))
        print(f"  Dataset too small for stratified split, using simple split: {len(train_idx)} train, {len(test_idx)} test")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    results = {}

    # ThinkARM (thinking only)
    print(f"\n  Testing ThinkARM (thinking episodes)...")
    X_train = train_df[ta_features].fillna(0).values
    X_test = test_df[ta_features].fillna(0).values
    y_train = train_df["correctness"].values
    y_test = test_df["correctness"].values

    results["ThinkARM (Thinking)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, np.array(ta_features), "ThinkARM (Thinking)"
    )

    # BLOOM (thinking only)
    print(f"  Testing BLOOM (thinking cognitive levels)...")
    X_train = train_df[bl_features].fillna(0).values
    X_test = test_df[bl_features].fillna(0).values

    results["BLOOM (Thinking)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, np.array(bl_features), "BLOOM (Thinking)"
    )

    # Combined (thinking only)
    print(f"  Testing Combined (both frameworks)...")
    all_features = ta_features + bl_features
    X_train = train_df[all_features].fillna(0).values
    X_test = test_df[all_features].fillna(0).values

    results["Combined (Thinking)"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, np.array(all_features), "Combined (Thinking)"
    )

    return results


def save_comparison_report(results, output_dir):
    """Save detailed comparison report."""
    report_file = Path(output_dir) / "thinking_comparison_report.txt"

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("THINKING TRACES ONLY - ThinkARM vs BLOOM for Correctness Prediction\n")
        f.write("=" * 80 + "\n\n")
        f.write("Note: Features extracted only from <think> blocks, excluding final answers.\n")
        f.write("This analyzes the reasoning process itself.\n\n")

        # Performance comparison
        f.write("1. PERFORMANCE COMPARISON (Test Set)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Framework':<30} {'Features':<12} {'Accuracy':<12} {'AUC':<12} {'F1':<12}\n")
        f.write("-" * 80 + "\n")

        for framework, res in results.items():
            f.write(
                f"{framework:<30} {res['n_features']:<12} {res['accuracy']:.4f}       "
                f"{res['auc']:.4f}       {res['f1']:.4f}\n"
            )

        # Detailed metrics
        f.write("\n\n2. DETAILED METRICS\n")
        f.write("-" * 80 + "\n")

        for framework, res in results.items():
            f.write(f"\n### {framework} ###\n")
            f.write(f"  Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"  Precision: {res['precision']:.4f}\n")
            f.write(f"  Recall:    {res['recall']:.4f}\n")
            f.write(f"  F1:        {res['f1']:.4f}\n")
            f.write(f"  AUC:       {res['auc']:.4f}\n")

        # Top features
        f.write("\n\n3. TOP PREDICTIVE FEATURES\n")
        f.write("-" * 80 + "\n")

        for framework, res in results.items():
            f.write(f"\n### {framework} ###\n")
            coefs = res["coefficients"].copy()

            # Positive features
            pos = coefs[coefs["coefficient"] > 0].sort_values("abs_coef", ascending=False).head(8)
            if len(pos) > 0:
                f.write(f"\n  Positive Contributors (Help Correctness):\n")
                for _, row in pos.iterrows():
                    f.write(f"    {row['feature']:<45} {row['coefficient']:>8.4f}\n")

            # Negative features
            neg = coefs[coefs["coefficient"] < 0].sort_values("abs_coef", ascending=False).head(8)
            if len(neg) > 0:
                f.write(f"\n  Negative Contributors (Hurt Correctness):\n")
                for _, row in neg.iterrows():
                    f.write(f"    {row['feature']:<45} {row['coefficient']:>8.4f}\n")

    print(f"\n✅ Report saved: {report_file}")


def main():
    args = parse_args()

    print("=" * 80)
    print("THINKING TRACES ONLY - ThinkARM vs BLOOM Comparison")
    print("=" * 80)

    # Load data
    print("\nLoading thinking traces...")
    df = load_thinking_data(args)

    if df is None or len(df) == 0:
        print("ERROR: No thinking traces loaded!")
        return

    # Compare frameworks
    results = compare_frameworks(df, args.output_dir)

    # Save report
    save_comparison_report(results, args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - Thinking Traces Only")
    print("=" * 80)
    for framework, res in results.items():
        print(f"\n{framework}:")
        print(f"  Features:  {res['n_features']}")
        print(f"  Accuracy:  {res['accuracy']:.4f}")
        print(f"  AUC:       {res['auc']:.4f}")
        print(f"  F1:        {res['f1']:.4f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Find best performer
    best_framework = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest F1 Score: {best_framework[0]} ({best_framework[1]['f1']:.4f})")

    ta_acc = results["ThinkARM (Thinking)"]['accuracy']
    bl_acc = results["BLOOM (Thinking)"]['accuracy']
    diff = abs(ta_acc - bl_acc)

    print(f"\nAccuracy Difference: {diff:.4f}")
    if abs(ta_acc - bl_acc) < 0.05:
        print("  → Frameworks perform similarly on thinking traces")
    else:
        winner = "ThinkARM" if ta_acc > bl_acc else "BLOOM"
        print(f"  → {winner} has notably better performance")

    print(f"\nFeatures used:")
    print(f"  ThinkARM: {results['ThinkARM (Thinking)']['n_features']} features")
    print(f"  BLOOM: {results['BLOOM (Thinking)']['n_features']} features")


if __name__ == "__main__":
    main()
