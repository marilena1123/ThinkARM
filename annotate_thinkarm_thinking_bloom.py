"""
Annotate ThinkARM thinking traces with Bloom's taxonomy — judge-segmented version
(Bloom-only; thinking traces only).

Adapted from run_cot_judge_v2.py to work with ThinkARM data format.
Extracts thinking traces from Response field and annotates them with Bloom taxonomy.

No correctness annotation is produced.

Output schema:
  thinking_bloom_labels — list of { step, bloom_level, bloom_reasoning }

Usage:
    python annotate_thinkarm_thinking_bloom.py \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --input_dir data/raw \
        --output_dir output_thinkarm_thinking_bloom \
        --models deepseekR1 "DeepSeek-R1-Distill-Qwen-1.5B" "DeepSeek-R1-Distill-Qwen-7B" Phi4R

Quick test (first 5 samples):
    python annotate_thinkarm_thinking_bloom.py \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --input_dir data/raw \
        --output_dir output_thinkarm_thinking_bloom \
        --models deepseekR1 \
        --max_samples 5
"""

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from tqdm import tqdm

BLOOM_LEVELS = [
    "REMEMBERING", "UNDERSTANDING", "APPLYING",
    "ANALYZING", "EVALUATING", "CREATING", "NONE",
]

BLOOM_JUDGE_VERSION = "v2_bloom_only_thinkarm"

JUDGE_SYSTEM_PROMPT = (
    "You are a data annotation expert. You segment reasoning traces "
    "into steps and classify each step according to Bloom's revised "
    "taxonomy."
)

JUDGE_USER_TEMPLATE = """You are a data annotation expert.

Given:
- A question
- A reasoning trace

Your task is to segment the reasoning trace into distinct steps, each aligned with exactly one cognitive level as defined by Bloom's Taxonomy.

## Segmentation and Labeling Rules
1. Segment by cognitive function, not by sentence. A segment can span multiple sentences or be a single clause — boundaries are determined by shifts in cognitive operation, not punctuation.
2. Start a new segment whenever the solver completes a cognitive operation (e.g., stops recalling and starts applying, or stops applying and starts verifying). This can include consecutive operations of the same type.
3. Transitional phrases ("Now...", "So...", "Let me...") belong to the segment of the operation they introduce, not the one before.
4. Higher taxonomy levels can subsume lower ones (e.g., Evaluate may involve Apply internally). In that case, label the segment by the highest level that characterizes the overall operation.
5. KEEP THE TEXT VERBATIM. DO NOT IMPROVE OR ALTER IT IN ANY WAY. DO NOT ADD text that is not present. Even when the reasoning is brief or incorrect, annotate it WITHOUT ADDITIONS and BASED ON WHAT IS PRESENT WITHOUT MAKING ANY ASSUMPTIONS ABOUT WHAT IS IMPLIED. OUR GOAL IS TO ANNOTATE THE EXISTING REASONING EXACTLY AS-IS. Do not omit any part of the reasoning trace.

## Bloom's Taxonomy Labels
Assign each segment exactly one label:

- REMEMBERING — Retrieves, recognizes, recalls or restates a fact, number, rule, or any information directly from the problem or prior knowledge, with no transformation or interpretation.
- UNDERSTANDING — Constructs meaning from information: interprets, infers, summarizes, compares, classifies, exemplifies, or explains what the information means or implies — going beyond repetition but not yet acting on it.
- APPLYING — Executes or implements a procedure, method, or algorithm.
- ANALYZING — Breaks the problem or information into parts, determines how those parts relate to each other and to the overall goal, or identifies dependencies between quantities and operations.
- EVALUATING — Makes a judgment based on criteria: checks whether a result is correct, verifies whether a method was valid, or critiques the quality of an approach.
- CREATING — Combining or synthesizing elements to produce a novel structure, formula, or solution strategy not derivable from a standard procedure. Assign only if a genuinely new path is invented.

## Question
{question}

## Reasoning Text to Segment and Classify
{reasoning_text}

## Output Format
One line per segment, in the order the reasoning proceeds.
Copy each segment's text verbatim — do not paraphrase or omit any part.
The concatenation of all step texts must exactly reconstruct the full reasoning text.

STEP 1: <step text> | <BLOOM_LEVEL> | BLOOM_REASON: <brief explanation>
STEP 2: <step text> | <BLOOM_LEVEL> | BLOOM_REASON: <brief explanation>
..."""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate ThinkARM thinking traces with Bloom's taxonomy labels "
                    "(judge-segmented, Bloom-only version)"
    )
    parser.add_argument("--judge_model_path", type=str, required=True,
                        help="Path to judge model (e.g. Llama-3.3-70B-Instruct)")
    parser.add_argument("--judge_name", type=str, default=None,
                        help="Short name for the judge model. Auto-detected if omitted.")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Directory with raw JSON files")
    parser.add_argument("--models", type=str, nargs='+',
                        default=["deepseekR1", "DeepSeek-R1-Distill-Qwen-1.5B",
                                "DeepSeek-R1-Distill-Qwen-7B", "Phi4R"],
                        help="Model names to process (without .json extension)")
    parser.add_argument("--output_dir", type=str, default="output_thinkarm_thinking_bloom",
                        help="Directory to save annotated JSONs")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per file (for quick testing)")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Max tokens for judge response")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Checkpoint every N samples")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of prompts to send to vLLM at once")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory for vLLM KV cache")
    args = parser.parse_args()

    if args.judge_name is None:
        model_dir = Path(args.judge_model_path).name.lower()
        if "prometheus" in model_dir:
            args.judge_name = "prometheus2"
        elif "llama" in model_dir:
            args.judge_name = "llama3_70b"
        else:
            args.judge_name = model_dir.replace("-", "_")
        print(f"Auto-detected judge name: {args.judge_name}")

    return args


BLOOM_ALIASES = {
    "REMEMBER": "REMEMBERING",
    "UNDERSTAND": "UNDERSTANDING",
    "APPLY": "APPLYING",
    "ANALYZE": "ANALYZING",
    "ANALYSE": "ANALYZING",
    "EVALUATE": "EVALUATING",
    "CREATE": "CREATING",
}


def parse_judge_response(response_text):
    """Parse the judge's response into per-step Bloom labels.

    Expected per-step format (one line per step, number of steps unknown):
        STEP N: <step text> | <BLOOM> | BLOOM_REASON: <explanation>

    Returns:
        list of dicts with keys: step, bloom_level, bloom_reasoning
    """
    all_level_names = list(BLOOM_LEVELS) + list(BLOOM_ALIASES.keys())
    levels_pattern = "|".join(all_level_names)

    # Discover how many steps the judge produced
    step_numbers = [int(n) for n in re.findall(r"STEP\s+(\d+)", response_text)]
    if not step_numbers:
        return []

    max_step = max(step_numbers)
    step_labels = []

    for i in range(1, max_step + 1):
        pattern = (
            rf"STEP\s+{i}(?!\d)\s*:\s*(.*?)"
            rf"\s*\|\s*\**({levels_pattern})\**"
            rf"\s*\|\s*\**BLOOM[_ ]REASON\**\s*:\s*(.*?)"
            rf"(?=\n\s*STEP\s+\d+\s*:|\Z)"
        )
        match = re.search(pattern, response_text,
                          re.IGNORECASE | re.DOTALL)
        if match:
            step_text = match.group(1).strip().strip('"').strip()
            level = match.group(2).strip().upper()
            bloom_reason = match.group(3).strip()
            level = BLOOM_ALIASES.get(level, level)
            if level not in BLOOM_LEVELS:
                level = "NONE"
            step_labels.append({
                "step": step_text,
                "bloom_level": level,
                "bloom_reasoning": bloom_reason,
            })
        else:
            step_labels.append({
                "step": "",
                "bloom_level": "PARSE_ERROR",
                "bloom_reasoning": "",
            })

    return step_labels


def extract_thinking_trace(response_text):
    """Extract thinking trace from Response field.

    Handles responses with <think>...</think> tags or similar markers.
    Returns the thinking portion only.
    """
    if not response_text:
        return ""

    # Try to extract from <think> tags
    if "<think>" in response_text:
        match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Otherwise return the whole response (assume it's all thinking)
    return response_text.strip()


def build_chat_prompt(tokenizer, question, reasoning_text, judge_name=""):
    """Build a chat-formatted prompt for the judge model."""
    user_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        reasoning_text=reasoning_text,
    )

    if "prometheus" in judge_name:
        return f"[INST] {JUDGE_SYSTEM_PROMPT}\n{user_content} [/INST]"

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def print_summary(label, all_labels):
    """Print Bloom level distribution across all steps."""
    total = len(all_labels)
    if total == 0:
        return

    bloom_counts = {lv: 0 for lv in BLOOM_LEVELS}
    bloom_counts["PARSE_ERROR"] = 0
    for lbl in all_labels:
        bl = lbl.get("bloom_level", "PARSE_ERROR")
        bloom_counts[bl] = bloom_counts.get(bl, 0) + 1

    print(f"\n  --- {label} ({total} steps) ---")
    print(f"  Bloom level distribution:")
    for lv in BLOOM_LEVELS + ["PARSE_ERROR"]:
        c = bloom_counts.get(lv, 0)
        if c > 0:
            print(f"    {lv:15s}: {c:>5} ({c/total:.1%})")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vLLM engine
    print(f"Loading judge model from {args.judge_model_path} with vLLM...")
    llm = LLM(
        model=args.judge_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    print("Judge model loaded.")

    # Process each model
    for model_name in args.models:
        input_file = input_dir / f"{model_name}.json"

        if not input_file.exists():
            print(f"\n[warn] File not found: {input_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {input_file.name}")
        print(f"{'='*60}")

        with open(input_file, "r") as f:
            data = json.load(f)

        # Handle both array and dict formats
        if isinstance(data, list):
            results = data
            summary = {"model": model_name}
        else:
            results = data.get("results", [])
            summary = data.get("summary", {"model": model_name})

        if args.max_samples is not None:
            results = results[:args.max_samples]
            print(f"  Limited to {args.max_samples} samples (test mode)")

        print(f"  {len(results)} samples from {model_name}")

        output_file = output_dir / input_file.name

        # Resume: load partially annotated file if it exists
        start_idx = 0
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [warn] Corrupt/empty checkpoint {output_file.name}, starting fresh")
                existing_data = {"results": []} if isinstance(data, dict) else []

            existing_results = existing_data.get("results", []) if isinstance(existing_data, dict) else existing_data

            # Count how many are fully annotated
            for i, r in enumerate(existing_results):
                has_thinking = "thinking_bloom_labels" in r or not r.get("Response", "").strip()
                if has_thinking:
                    start_idx = i + 1
                else:
                    break
            if start_idx > 0:
                print(f"  Resuming from sample {start_idx}")
                for i in range(start_idx):
                    if i < len(results):
                        results[i] = existing_results[i]

        if start_idx >= len(results):
            print(f"  All samples already annotated for {input_file.name}")
            continue

        # Collect all (idx, thinking_text) tuples that need annotation
        to_annotate = []
        for idx in range(start_idx, len(results)):
            r = results[idx]

            # Extract thinking trace from Response field
            response = r.get("Response") or ""
            thinking = extract_thinking_trace(response)

            if thinking.strip() and "thinking_bloom_labels" not in r:
                to_annotate.append((idx, thinking))

        print(f"  {len(to_annotate)} thinking traces to annotate")

        if not to_annotate:
            print(f"  No thinking traces to annotate")
            continue

        # Build prompts
        prompts = []
        for idx, thinking_text in to_annotate:
            r = results[idx]
            question = r.get("Instruction", r.get("question", ""))
            prompts.append(build_chat_prompt(
                tokenizer, question, thinking_text,
                judge_name=args.judge_name,
            ))

        all_labels = []

        # Process in batches
        batch_size = args.batch_size
        for batch_start in tqdm(range(0, len(prompts), batch_size),
                                desc=model_name):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_items = to_annotate[batch_start:batch_end]

            outputs = llm.generate(batch_prompts, sampling_params)

            for output, (idx, thinking_text) in zip(outputs, batch_items):
                judge_response = output.outputs[0].text.strip()
                step_labels = parse_judge_response(judge_response)

                results[idx]["thinking_bloom_labels"] = step_labels
                results[idx]["bloom_judge"] = args.judge_name
                results[idx]["bloom_judge_version"] = BLOOM_JUDGE_VERSION
                results[idx]["thinking_bloom_judge_raw"] = judge_response
                all_labels.extend(step_labels)

            # Checkpoint
            total_done = batch_end
            if total_done % args.save_every < batch_size or batch_end == len(prompts):
                if isinstance(data, dict):
                    out_data = {**data, "results": results}
                    out_data.setdefault("summary", {})["bloom_judge"] = args.judge_name
                    out_data["summary"]["bloom_judge_version"] = BLOOM_JUDGE_VERSION
                else:
                    out_data = results

                with open(output_file, "w") as f:
                    json.dump(out_data, f, indent=2)

        # Final save
        if isinstance(data, dict):
            out_data = {**data, "results": results}
            out_data.setdefault("summary", {})["bloom_judge"] = args.judge_name
            out_data["summary"]["bloom_judge_version"] = BLOOM_JUDGE_VERSION
        else:
            out_data = results

        with open(output_file, "w") as f:
            json.dump(out_data, f, indent=2)

        print_summary(model_name, all_labels)

    print("\nDone. Annotated files saved to", output_dir)


if __name__ == "__main__":
    main()
