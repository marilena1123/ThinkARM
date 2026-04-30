"""
Utilities for organizing and analyzing generated and annotated data.

Functions for:
- Merging multiple results
- Computing statistics
- Filtering by criteria
- Converting between formats
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import argparse


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []


def save_results(results: List[Dict[str, Any]], path: str, summary: Dict = None):
    """Save results to a JSON file."""
    output_data = {
        "summary": summary or {"_summary": True},
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output_data, f, indent=2)


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy statistics."""
    total = len(results)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    correct = sum(1 for r in results if r.get("correct", False))

    thinking_correct = 0
    thinking_available = 0

    for r in results:
        if "thinking_correct" in r:
            thinking_available += 1
            if r["thinking_correct"]:
                thinking_correct += 1

    stats = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
    }

    if thinking_available > 0:
        stats["thinking_correct"] = thinking_correct
        stats["thinking_available"] = thinking_available
        stats["thinking_accuracy"] = thinking_correct / thinking_available

    return stats


def get_episode_distribution(results: List[Dict[str, Any]], use_thinking: bool = False) -> Dict[str, int]:
    """Get distribution of ThinkARM episode categories."""
    episodes = Counter()

    for result in results:
        label_key = "thinking_thinkarm_labels" if use_thinking else "thinkarm_labels"
        labels = result.get(label_key)

        if labels:
            for label in labels:
                if isinstance(label, dict) and "category" in label:
                    episodes[label["category"]] += 1

    return dict(episodes)


def filter_by_correctness(results: List[Dict[str, Any]], correct: bool = True) -> List[Dict[str, Any]]:
    """Filter results by correctness."""
    return [r for r in results if r.get("correct") == correct]


def filter_by_task(results: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """Filter results by task type."""
    return [r for r in results if r.get("task") == task]


def get_episode_accuracy_correlation(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Analyze correlation between episode distribution and correctness."""
    correct_results = filter_by_correctness(results, correct=True)
    incorrect_results = filter_by_correctness(results, correct=False)

    correct_episodes = get_episode_distribution(correct_results)
    incorrect_episodes = get_episode_distribution(incorrect_results)

    # Normalize by count
    total_correct = sum(correct_episodes.values())
    total_incorrect = sum(incorrect_episodes.values())

    correct_dist = {
        ep: count / total_correct if total_correct > 0 else 0
        for ep, count in correct_episodes.items()
    }
    incorrect_dist = {
        ep: count / total_incorrect if total_incorrect > 0 else 0
        for ep, count in incorrect_episodes.items()
    }

    return {
        "correct": correct_dist,
        "incorrect": incorrect_dist,
    }


def merge_results(*file_paths: str, output_path: str = None) -> List[Dict[str, Any]]:
    """Merge multiple results files."""
    merged = []

    for path in file_paths:
        results = load_results(path)
        merged.extend(results)

    if output_path:
        summary = compute_accuracy(merged)
        save_results(merged, output_path, summary)

    return merged


def print_statistics(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("GENERATION STATISTICS")
    print("="*60)

    accuracy_stats = compute_accuracy(results)
    print(f"Total results: {accuracy_stats['total']}")
    print(f"Correct: {accuracy_stats['correct']}")
    print(f"Accuracy: {accuracy_stats['accuracy']:.4f}")

    if "thinking_accuracy" in accuracy_stats:
        print(f"\nThinking Trace Statistics:")
        print(f"  Thinking available: {accuracy_stats['thinking_available']}")
        print(f"  Thinking correct: {accuracy_stats['thinking_correct']}")
        print(f"  Thinking accuracy: {accuracy_stats['thinking_accuracy']:.4f}")

    # Episode distribution
    episode_dist = get_episode_distribution(results)
    if episode_dist:
        print(f"\nEpisode Distribution (Final Reasoning):")
        total_episodes = sum(episode_dist.values())
        for episode, count in sorted(episode_dist.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_episodes
            print(f"  {episode}: {count} ({pct:.1f}%)")

    # Thinking episode distribution
    thinking_dist = get_episode_distribution(results, use_thinking=True)
    if thinking_dist:
        print(f"\nEpisode Distribution (Thinking Traces):")
        total_episodes = sum(thinking_dist.values())
        for episode, count in sorted(thinking_dist.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_episodes
            print(f"  {episode}: {count} ({pct:.1f}%)")

    # Correlation analysis
    print(f"\nEpisode-Correctness Correlation:")
    correlation = get_episode_accuracy_correlation(results)

    print(f"  Correct solutions:")
    for episode, pct in sorted(correlation["correct"].items(), key=lambda x: -x[1]):
        if pct > 0:
            print(f"    {episode}: {pct:.1%}")

    print(f"  Incorrect solutions:")
    for episode, pct in sorted(correlation["incorrect"].items(), key=lambda x: -x[1]):
        if pct > 0:
            print(f"    {episode}: {pct:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Data utilities for ThinkARM analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Print statistics")
    stats_parser.add_argument("--path", required=True, help="Path to results file")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge results files")
    merge_parser.add_argument("--inputs", nargs="+", required=True, help="Input files to merge")
    merge_parser.add_argument("--output", required=True, help="Output file")

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter results")
    filter_parser.add_argument("--path", required=True, help="Input file")
    filter_parser.add_argument("--output", required=True, help="Output file")
    filter_parser.add_argument("--correct", action="store_true", help="Keep only correct results")
    filter_parser.add_argument("--incorrect", action="store_true", help="Keep only incorrect results")
    filter_parser.add_argument("--task", help="Filter by task type")

    args = parser.parse_args()

    if args.command == "stats":
        results = load_results(args.path)
        print_statistics(results)

    elif args.command == "merge":
        merged = merge_results(*args.inputs, output_path=args.output)
        print(f"Merged {len(merged)} results to {args.output}")
        print_statistics(merged)

    elif args.command == "filter":
        results = load_results(args.path)

        if args.correct:
            results = filter_by_correctness(results, correct=True)
        elif args.incorrect:
            results = filter_by_correctness(results, correct=False)

        if args.task:
            results = filter_by_task(results, args.task)

        save_results(results, args.output)
        print(f"Filtered to {len(results)} results, saved to {args.output}")
        print_statistics(results)


if __name__ == "__main__":
    main()
