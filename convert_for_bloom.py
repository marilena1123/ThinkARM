"""
Convert generated thinking traces to BLOOM annotation format.

Takes output from generate.py and formats it for input to the BLOOM
annotation script (annotate_bloom.py), combining thinking_steps back
into full thinking traces and preparing reasoning for annotation.

Usage:
    python convert_for_bloom.py \
        --input outputs/gen_qwen3_30b_thinking_gsm8k.json \
        --output outputs/bloom_input_qwen3_30b_thinking_gsm8k.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert generated traces to BLOOM annotation format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to generated traces from generate.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save BLOOM-formatted output",
    )
    return parser.parse_args()


def reconstruct_thinking_trace(thinking_steps: List[str]) -> str:
    """Reconstruct full thinking trace from steps."""
    if not thinking_steps:
        return ""

    # Join steps with space (they're already sentence-like)
    return " ".join(thinking_steps)


def reconstruct_reasoning(steps: List[str]) -> str:
    """Reconstruct full reasoning from steps."""
    if not steps:
        return ""

    return " ".join(steps)


def convert_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single result to BLOOM format."""

    converted = {
        "question": result.get("question", ""),
        "predicted_answer": result.get("predicted_answer"),
        "ground_truth": result.get("ground_truth"),
        "correct": result.get("correct", False),
    }

    # Add task if present
    if "task" in result:
        converted["task"] = result["task"]

    # Reconstruct and add reasoning
    if result.get("steps"):
        converted["reasoning"] = reconstruct_reasoning(result["steps"])
    else:
        converted["reasoning"] = result.get("reasoning", "")

    # Reconstruct and add thinking trace
    if result.get("thinking_steps"):
        converted["thinking_trace"] = reconstruct_thinking_trace(result["thinking_steps"])
    elif result.get("thinking_trace"):
        converted["thinking_trace"] = result["thinking_trace"]

    # Preserve original fields for reference
    converted["_original"] = {
        "steps": result.get("steps"),
        "thinking_steps": result.get("thinking_steps"),
    }

    return converted


def main():
    args = parse_args()

    # Load input
    print(f"Loading generated traces from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        summary = data.get("summary", {})
    else:
        results = data
        summary = {"_summary": True}

    print(f"Loaded {len(results)} results")

    # Convert results
    print("Converting to BLOOM format...")
    converted_results = []

    for i, result in enumerate(results):
        converted = convert_result(result)
        converted_results.append(converted)

        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{len(results)}")

    # Build output
    output_data = {
        "summary": summary,
        "results": converted_results,
    }

    # Save output
    print(f"Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone! Converted {len(converted_results)} results")
    print(f"Ready for BLOOM annotation: {args.output}")

    # Show sample
    if converted_results:
        print("\nSample result:")
        sample = converted_results[0]
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Reasoning length: {len(sample.get('reasoning', ''))} chars")
        print(f"  Thinking length: {len(sample.get('thinking_trace', ''))} chars")


if __name__ == "__main__":
    main()
