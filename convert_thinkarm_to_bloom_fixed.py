"""
Convert ThinkARM annotated data to BLOOM annotation format (FIXED).

Fixed version that handles the actual data structure:
- Raw data uses "Instruction" (not "question") and "Correct Answer" (not "answer")
- Index matching between label and raw files
- Proper metadata extraction
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ThinkARM annotated data to BLOOM format (fixed)"
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
        help="Path to ThinkARM data/raw directory",
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
        help="Comma-separated models (e.g. deepseekR1,Phi4R). If None, converts all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max problems per model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="math",
        help="Dataset name for metadata",
    )
    return parser.parse_args()


def load_raw_data(raw_file: Path) -> Dict[int, Dict[str, Any]]:
    """Load raw data from a single JSON file."""
    raw_data = {}

    try:
        with open(raw_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                # Use the "index" field if available, otherwise use position
                idx = item.get("index")
                if idx is not None:
                    raw_data[idx] = item

        return raw_data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {raw_file}: {e}")
        return {}


def reconstruct_trace(sentences: List[Dict[str, Any]], sentence_type: str = "think") -> str:
    """Reconstruct full trace from annotated sentences."""
    matching = [
        s["sentence"]
        for s in sentences
        if s.get("sentence-type") == sentence_type
    ]
    return " ".join(matching) if matching else ""


def preserve_thinkarm_annotations(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve original ThinkARM annotations."""
    return [
        {
            "index": s.get("index"),
            "sentence": s.get("sentence"),
            "sentence-type": s.get("sentence-type"),
            "sentence-category": s.get("sentence-category"),
            "sentence-category-reason": s.get("sentence-category-reason"),
        }
        for s in sentences
    ]


def convert_model(
    model_name: str,
    thinkarm_dir: Path,
    raw_data: Dict[int, Dict[str, Any]],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert all problems for a single model."""
    results = []
    model_dir = thinkarm_dir / model_name

    if not model_dir.exists():
        print(f"  Model dir not found: {model_dir}")
        return results

    # Find all problem JSON files (numbered like 1.json, 2.json, etc.)
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

        # Match with raw data using problem number (1-indexed)
        problem_num = int(problem_file.stem)
        raw = raw_data.get(problem_num)

        if not raw:
            print(f"    Warning: No raw data for problem {problem_num}")
            continue

        # Extract from raw data using correct field names
        question = raw.get("Instruction", "")
        answer = raw.get("Correct Answer", "")

        if not question:
            continue

        # Reconstruct traces
        thinking_trace = reconstruct_trace(sentences, "think")
        answer_trace = reconstruct_trace(sentences, "answer")
        full_trace = " ".join([thinking_trace, answer_trace]).strip()

        if not full_trace:
            continue

        result = {
            "question": question,
            "reasoning": full_trace,
            "thinking_trace": thinking_trace,
            "answer_trace": answer_trace,
            "ground_truth": answer,
            "model": model_name,
            "problem_id": problem_num,
            "thinkarm_original_sentences": preserve_thinkarm_annotations(sentences),
        }

        # Add optional fields from raw data
        if "domain" in raw:
            result["domain"] = raw["domain"]
        if "difficulty" in raw:
            result["difficulty"] = raw["difficulty"]
        if "source" in raw:
            result["source"] = raw["source"]

        results.append(result)

    return results


def main():
    args = parse_args()

    thinkarm_dir = Path(args.thinkarm_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not thinkarm_dir.exists():
        print(f"ERROR: ThinkARM dir not found: {thinkarm_dir}")
        sys.exit(1)

    if not raw_dir.exists():
        print(f"ERROR: Raw dir not found: {raw_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models to process
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = sorted([d.name for d in thinkarm_dir.iterdir() if d.is_dir()])

    print(f"Found {len(models)} models: {', '.join(models)}")

    # Find raw data files
    raw_files = {}
    for raw_file in raw_dir.glob("*.json"):
        # Map file to model name (e.g., deepseekR1.json -> deepseekR1)
        model_key = raw_file.stem
        raw_files[model_key] = raw_file

    print(f"Found {len(raw_files)} raw data files")

    # Convert each model
    for model_name in models:
        print(f"\nConverting {model_name}...")

        # Find matching raw file
        raw_file = raw_files.get(model_name)
        if not raw_file:
            print(f"  No raw data file for {model_name}")
            continue

        # Load raw data
        raw_data = load_raw_data(raw_file)
        print(f"  Loaded {len(raw_data)} raw samples")

        # Convert
        results = convert_model(
            model_name,
            thinkarm_dir,
            raw_data,
            max_samples=args.max_samples,
        )

        if not results:
            print(f"  No results converted for {model_name}")
            continue

        print(f"  Converted {len(results)} problems")

        # Save
        output_file = output_dir / f"bloom_input_{model_name}_{args.dataset_name}.json"

        summary = {
            "model": model_name,
            "dataset": args.dataset_name,
            "total": len(results),
            "source": "ThinkARM annotated data",
        }

        output_data = {
            "summary": summary,
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"  Saved to {output_file}")

    print(f"\nDone! Output in {output_dir}")
    print(f"\nNext: python annotate_bloom.py \\")
    print(f"  --judge_model_path /path/to/Llama-3.3-70B-Instruct \\")
    print(f"  --input_dir {output_dir} \\")
    print(f"  --pattern 'bloom_input_*.json'")


if __name__ == "__main__":
    main()
