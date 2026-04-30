"""
Annotate thinking traces with ThinkARM episode labels using a local judge model.

Uses the ThinkARM framework to label reasoning traces into episode categories:
Read, Analyze, Explore, Plan, Implement, Verify, Monitor, Answer.

Usage:
    python annotate_thinking_traces.py \
        --input_path ./outputs/gen_model_dataset.json \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --output_dir ./outputs_annotated \
        --batch_size 8
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re

from vllm import LLM, SamplingParams
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate thinking traces with ThinkARM episode labels"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to generated traces JSON",
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        required=True,
        help="Path to local Llama judge model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_annotated",
        help="Directory to save annotated results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for annotation",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM KV cache",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index within the results",
    )
    parser.add_argument(
        "--annotate_thinking_only",
        action="store_true",
        help="Only annotate thinking traces, not final reasoning",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N examples",
    )
    return parser.parse_args()


# ThinkARM episode categories
EPISODE_CATEGORIES = {
    "Read": "Reading and understanding the problem statement",
    "Analyze": "Analyzing the problem structure and identifying key elements",
    "Explore": "Exploring different approaches and strategies",
    "Plan": "Planning the solution approach",
    "Implement": "Implementing the solution (concrete execution)",
    "Verify": "Verifying the solution for correctness",
    "Monitor": "Monitoring progress and evaluating intermediate results",
    "Answer": "Providing the final answer",
}

GUIDEBOOK = f"""# ThinkARM Episode Categories

The following are the eight episode categories for analyzing reasoning:

**Read**: Reading and understanding the problem statement. The solver identifies key information, constraints, and goals from the problem.

**Analyze**: Analyzing the problem structure and identifying key elements. The solver breaks down the problem, identifies relevant concepts, and establishes relationships.

**Explore**: Exploring different approaches and strategies. The solver considers multiple ways to solve the problem, tests hypotheses, and investigates possibilities.

**Plan**: Planning the solution approach. The solver outlines the strategy, defines steps, and prepares for implementation.

**Implement**: Implementing the solution through concrete execution. The solver performs calculations, applies algorithms, and carries out the planned steps.

**Verify**: Verifying the solution for correctness. The solver checks work, validates assumptions, and ensures accuracy.

**Monitor**: Monitoring progress and evaluating intermediate results. The solver assesses whether the approach is working, adjusts course, or provides feedback.

**Answer**: Providing the final answer. The solver states the conclusion or solution.
"""


def build_annotation_prompt(steps: List[str], guidebook: bool = True) -> str:
    """Build a prompt for annotating steps with ThinkARM categories."""

    formatted_steps = "\n".join([f"[{i+1}] {step}" for i, step in enumerate(steps)])

    prompt = f"""You are an expert in analyzing reasoning processes using Schoenfeld's Episode Theory framework.

Your task is to analyze the following reasoning steps and classify each into one of these 8 episode categories:
- Read: Understanding the problem
- Analyze: Analyzing problem structure
- Explore: Exploring different approaches
- Plan: Planning the solution
- Implement: Concrete execution
- Verify: Verifying correctness
- Monitor: Evaluating progress
- Answer: Providing final answer

{GUIDEBOOK if guidebook else ""}

Steps to annotate:
{formatted_steps}

For each step, provide:
1. The step index
2. The episode category (one of: Read, Analyze, Explore, Plan, Implement, Verify, Monitor, Answer)
3. A brief reason for the classification

Format your response as JSON with this structure:
{{
  "annotations": [
    {{"index": 1, "category": "Read", "reason": "..."}},
    {{"index": 2, "category": "Analyze", "reason": "..."}}
  ]
}}

Analyze the steps carefully and provide accurate classifications:"""

    return prompt


def parse_annotation_response(response_text: str, num_steps: int) -> List[Dict[str, Any]]:
    """Parse the judge model's response to extract annotations."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            response_json = json.loads(json_match.group())
            annotations = response_json.get("annotations", [])
            return annotations
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: create empty annotations
    return [
        {"index": i, "category": "Unknown", "reason": "Failed to parse"}
        for i in range(1, num_steps + 1)
    ]


def annotate_steps(
    llm: LLM,
    steps: List[str],
    sampling_params: SamplingParams,
    tokenizer,
) -> List[Dict[str, Any]]:
    """Annotate a list of steps using the judge model."""

    if not steps:
        return []

    prompt = build_annotation_prompt(steps, guidebook=True)

    # Apply chat template
    messages = [
        {"role": "system", "content": "You are a reasoning analysis expert using Schoenfeld's Episode Theory."},
        {"role": "user", "content": prompt}
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate response
    outputs = llm.generate([chat_prompt], sampling_params)
    response_text = outputs[0].outputs[0].text.strip()

    # Parse response
    annotations = parse_annotation_response(response_text, len(steps))

    return {
        "annotations": annotations,
        "raw_response": response_text
    }


def process_result(
    result: Dict[str, Any],
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    annotate_thinking_only: bool = False
) -> Dict[str, Any]:
    """Process a single generated result and add annotations."""

    # Annotate final reasoning steps
    result["thinkarm_labels"] = None
    result["thinkarm_judge"] = "llama3_70b_instruct"
    result["thinkarm_judge_version"] = "v1_thinkarm_vllm"
    result["thinkarm_judge_raw"] = None

    if not annotate_thinking_only and result.get("steps"):
        annotation_result = annotate_steps(
            llm, result["steps"], sampling_params, tokenizer
        )
        result["thinkarm_labels"] = annotation_result["annotations"]
        result["thinkarm_judge_raw"] = annotation_result["raw_response"]

    # Annotate thinking traces if available
    result["thinking_thinkarm_labels"] = None

    if result.get("thinking_steps"):
        annotation_result = annotate_steps(
            llm, result["thinking_steps"], sampling_params, tokenizer
        )
        result["thinking_thinkarm_labels"] = annotation_result["annotations"]

    return result


def main():
    args = parse_args()

    # Load input data
    print(f"Loading generated traces from {args.input_path}...")
    with open(args.input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        summary = data.get("summary", {})
    else:
        results = data
        summary = {}

    results = results[args.start_idx:]
    print(f"Processing {len(results)} results for annotation")

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_filename = Path(args.input_path).stem
    output_filename = f"{input_filename}_annotated.json"
    output_path = output_dir / output_filename

    # Load existing annotations if resuming
    annotated_results = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        if isinstance(existing, dict) and "results" in existing:
            annotated_results = existing["results"]
        else:
            annotated_results = existing
        print(f"Loaded {len(annotated_results)} existing annotations, resuming...")
        results = results[len(annotated_results):]

    if not results:
        print("All results already annotated.")
        return

    # Initialize judge model
    print(f"Loading judge model from {args.judge_model_path}...")
    llm = LLM(
        model=args.judge_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=2048,
    )

    tokenizer = llm.get_tokenizer()

    # Process in batches
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(results), batch_size), desc="Annotating"):
        batch_end = min(batch_start + batch_size, len(results))
        batch = results[batch_start:batch_end]

        for result in batch:
            annotated = process_result(
                result, llm, sampling_params, tokenizer,
                annotate_thinking_only=args.annotate_thinking_only
            )
            annotated_results.append(annotated)

        # Checkpoint
        total_so_far = len(annotated_results)
        if total_so_far % args.save_every < batch_size or batch_end == len(results):
            print(f"  [{total_so_far}] annotated so far")
            output_data = {
                "summary": summary,
                "results": annotated_results,
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

    # Final save
    output_data = {
        "summary": summary,
        "results": annotated_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone. Annotated {len(annotated_results)} results")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
