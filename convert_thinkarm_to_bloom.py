"""
Convert ThinkARM annotated data to BLOOM annotation format.

Takes ThinkARM's annotated sentence-level data and:
1. Reconstructs the full reasoning trace from sentences
2. Formats it for input to BLOOM annotator
3. Preserves metadata for later comparison

Input format (ThinkARM):
  [
    {
      "index": 1,
      "sentence": "...",
      "sentence-type": "think" or "answer",
      "sentence-category": "Read|Analyze|Explore|...",
      "sentence-category-reason": "..."
    },
    ...
  ]

Output format (for BLOOM):
  {
    "summary": {...},
    "results": [
      {
        "question": "...",
        "reasoning": "full reconstructed trace",
        "thinking_trace": "...",
        "thinkarm_labels": [...],  # Preserve original annotations
        "ground_truth": "...",
        "correct": true/false
      }
    ]
  }

Usage:
    python convert_thinkarm_to_bloom.py \
        --thinkarm_dir /path/to/ThinkARM/data/label \
        --raw_dir /path/to/ThinkARM/data/raw \
        --output_dir ./outputs_bloom_from_thinkarm \
        --models QwQ32B,deepseekR1 \
        --max_samples 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ThinkARM annotated data to BLOOM format"
    )
    parser.add_argument(
        "--thinkarm_dir",
        type=str,
        required=True,
        help="Path to ThinkARM data/label directory",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to ThinkARM data/raw directory (for metadata)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_bloom_from_thinkarm",
        help="Directory to save BLOOM-formatted data",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to convert (e.g. QwQ32B,deepseekR1). If None, converts all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per model to process",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="math",
        help="Dataset name for metadata",
    )
    return parser.parse_args()


def load_raw_data(raw_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load raw model outputs for metadata (questions, answers)."""
    raw_data = {}

    # Find all raw JSON files
    raw_files = list(raw_dir.glob("*.json"))

    for raw_file in raw_files:
        try:
            with open(raw_file, "r") as f:
                data = json.load(f)

            # Data might be list or dict depending on format
            if isinstance(data, list):
                for item in data:
                    # Use problem_id or index as key
                    problem_id = item.get("problem_id") or item.get("index") or len(raw_data)
                    raw_data[problem_id] = item
            elif isinstance(data, dict):
                raw_data.update(data)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {raw_file}: {e}")

    return raw_data


def reconstruct_trace(sentences: List[Dict[str, Any]], sentence_type: str = "think") -> str:
    """Reconstruct full trace from annotated sentences of a given type."""
    matching = [
        s["sentence"]
        for s in sentences
        if s.get("sentence-type") == sentence_type
    ]

    return " ".join(matching) if matching else ""


def preserve_thinkarm_annotations(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve original ThinkARM annotations for comparison."""
    preserved = []

    for sent in sentences:
        preserved.append({
            "index": sent.get("index"),
            "sentence": sent.get("sentence"),
            "sentence-type": sent.get("sentence-type"),
            "sentence-category": sent.get("sentence-category"),
            "sentence-category-reason": sent.get("sentence-category-reason"),
        })

    return preserved


def convert_model(
    model_name: str,
    thinkarm_dir: Path,
    raw_data: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert all problems for a single model."""
    results = []

    model_dir = thinkarm_dir / model_name

    if not model_dir.exists():
        print(f"  Warning: Model directory not found: {model_dir}")
        return results

    # Find all problem JSON files
    problem_files = sorted(model_dir.glob("*.json"), key=lambda p: int(p.stem))

    for i, problem_file in enumerate(problem_files):
        if max_samples and i >= max_samples:
            break

        try:
            with open(problem_file, "r") as f:
                sentences = json.load(f)

        except (json.JSONDecodeError, IOError) as e:
            print(f"    Warning: Could not load {problem_file}: {e}")
            continue

        if not isinstance(sentences, list):
            print(f"    Warning: Expected list in {problem_file}")
            continue

        # Get problem metadata from raw data
        problem_id = int(problem_file.stem)
        raw = raw_data.get(problem_id - 1) or raw_data.get(problem_id) or {}

        question = raw.get("question", "")
        answer = raw.get("answer") or raw.get("ground_truth", "")
        correct = raw.get("correct")

        # Separate thinking and answer sentences
        thinking_sentences = [s for s in sentences if s.get("sentence-type") == "think"]
        answer_sentences = [s for s in sentences if s.get("sentence-type") == "answer"]

        # Reconstruct traces
        thinking_trace = reconstruct_trace(sentences, "think")
        answer_trace = reconstruct_trace(sentences, "answer")

        # For reasoning, combine both (thinking + answer)
        full_trace = " ".join([thinking_trace, answer_trace]).strip()

        if not full_trace or not question:
            continue

        result = {
            "question": question,
            "reasoning": full_trace,
            "thinking_trace": thinking_trace,
            "answer_trace": answer_trace,
            "ground_truth": answer,
            "correct": correct if correct is not None else None,
            "model": model_name,
            "problem_id": problem_id,
            # Preserve original ThinkARM annotations for later comparison
            "thinkarm_original_sentences": preserve_thinkarm_annotations(sentences),
        }

        results.append(result)

    return results


def main():
    args = parse_args()

    thinkarm_dir = Path(args.thinkarm_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # Validate directories
    if not thinkarm_dir.exists():
        print(f"ERROR: ThinkARM directory not found: {thinkarm_dir}")
        sys.exit(1)

    if not raw_dir.exists():
        print(f"ERROR: Raw directory not found: {raw_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to process
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        # Auto-detect models from subdirectories
        models = [d.name for d in thinkarm_dir.iterdir() if d.is_dir()]

    models = sorted(models)
    print(f"Found {len(models)} models: {', '.join(models)}")

    # Load raw data once
    print("Loading raw data...")
    raw_data = load_raw_data(raw_dir)
    print(f"Loaded metadata for {len(raw_data)} problems")

    # Convert each model
    for model_name in models:
        print(f"\nConverting {model_name}...")

        results = convert_model(
            model_name,
            thinkarm_dir,
            raw_data,
            max_samples=args.max_samples,
        )

        if not results:
            print(f"  No results for {model_name}")
            continue

        print(f"  Converted {len(results)} problems")

        # Save
        output_file = output_dir / f"bloom_input_{model_name}_{args.dataset_name}.json"

        summary = {
            "model": model_name,
            "dataset": args.dataset_name,
            "total": len(results),
            "source": "ThinkARM annotated data",
            "_conversion_note": "Full reasoning reconstructed from ThinkARM sentence annotations",
        }

        output_data = {
            "summary": summary,
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"  Saved to {output_file}")

    print(f"\nDone! Output saved to {output_dir}")
    print(f"\nNext step: Feed to BLOOM annotator")
    print(f"  python annotate_bloom.py \\")
    print(f"    --judge_model_path /path/to/Llama-3.3-70B-Instruct \\")
    print(f"    --input_dir {output_dir} \\")
    print(f"    --pattern 'bloom_input_*.json' \\")
    print(f"    --output_dir outputs_bloom_annotated")


if __name__ == "__main__":
    main()
