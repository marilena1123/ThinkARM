#!/usr/bin/env python3
"""
Annotate CoT reasoning with Bloom's taxonomy — judge-segmented version
(Bloom-only; correctness annotation removed).

Local vLLM version adjusted to match the OpenRouter version as closely as possible:
  - temperature CLI arg, default 0.0
  - max_tokens default 8192
  - top_p explicitly set to 1.0
  - Bloom labels use short forms: REMEMBER, UNDERSTAND, APPLY, ANALYZE, EVALUATE, CREATE, NONE
  - parser accepts both short and -ING variants, then normalizes to short forms
"""

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from tqdm import tqdm

BLOOM_LEVELS = [
    "REMEMBER", "UNDERSTAND", "APPLY",
    "ANALYZE", "EVALUATE", "CREATE", "NONE",
]

BLOOM_JUDGE_VERSION = "v2_bloom_only_vllm_matched"

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
2. Start a new segment whenever the model switches cognitive operation (e.g., stops recalling and starts applying, or stops applying and starts verifying).
3. Transitional phrases ("Now...", "So...", "Let me...") belong to the segment of the operation they introduce, not the one before.
4. Higher taxonomy levels can subsume lower ones (e.g., Evaluate may involve Apply internally). In that case, label the segment by the highest level that characterizes the overall operation.
5. KEEP THE TEXT VERBATIM. DO NOT IMPROVE OR ALTER IT IN ANY WAY. DO NOT ADD text that is not present. Even when the reasoning is brief, annotate is WITHOUT ADDITIONS and BASED ON WHAT IS PRESENT WITHOUT MAKING ANY ASSUMPTIONS ABOUT WHAT IS IMPLIED. OUR GOAL IS TO ANNOTATE THE EXISTING REASONING EXACTLY AS-IS. Do not omit any part of the reasoning trace.

## Bloom's Taxonomy Labels
Assign each segment exactly one label:

- REMEMBER — Retrieves or restates a fact, number, rule, or any information directly from the problem or prior knowledge, with no transformation or interpretation.
- UNDERSTAND — Constructs meaning from information: interprets, infers, summarizes, compares, classifies, exemplifies, or explains what the information means or implies — going beyond repetition but not yet acting on it.
- APPLY — Executes a procedure, method, or algorithm.
- ANALYZE — Breaks the problem or information into parts, determines how those parts relate to each other and to the overall goal, or identifies dependencies between quantities and operations.
- EVALUATE — Makes a judgment based on criteria: checks whether a result is correct, verifies whether a method was valid, or critiques the quality of an approach.
- CREATE — Generates a novel structure, formula, or solution strategy not derivable from a standard procedure. Assign only if a genuinely new path is invented.

## Worked Examples

#Example 1
Question:
Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

Reasoning trace:
Okay, let's tackle this problem step by step. First, I need to figure out how many pages Julie has read so far and then determine how many are left. The book is 120 pages total. Yesterday, she read 12 pages. Today, she read twice as many as yesterday. So, today's pages would be 2 times 12, which is 24 pages. Let me add those up: 12 + 24 = 36 pages read in total over the two days. Now, subtract that from the total pages to find the remaining pages. 120 minus 36 equals 84 pages left. The question says she wants to read half of the remaining pages tomorrow. So, half of 84 is 42. Therefore, she should read 42 pages tomorrow. Let me double-check to make sure I didn't make a mistake. Total pages: 120. Yesterday: 12. Today: 24. Total read: 36. Remaining: 84. Half of 84 is 42. Yep, that seems right.

Correct segmentation:
STEP 1: Okay, let's tackle this problem step by step. First, I need to figure out how many pages Julie has read so far and then determine how many are left. | ANALYZE | BLOOM_REASON: The solver breaks the overall task into subproblems (finding pages read so far and pages remaining) and identifies how these parts relate to the goal of solving the problem.
STEP 2: The book is 120 pages total. Yesterday, she read 12 pages. Today, she read twice as many as yesterday. | REMEMBER | BLOOM_REASON: The solver restates numerical facts given in the problem (total pages, pages read yesterday, and the relationship for today) without any transformation or interpretation.
STEP 3: So, today's pages would be 2 times 12, which is 24 pages. Let me add those up: 12 + 24 = 36 pages read in total over the two days. Now, subtract that from the total pages to find the remaining pages. 120 minus 36 equals 84 pages left. | APPLY | BLOOM_REASON: The solver executes a sequence of arithmetic procedures (multiplication, addition, and subtraction) to compute pages read so far and the number of remaining pages.
STEP 4: The question says she wants to read half of the remaining pages tomorrow. | REMEMBER | BLOOM_REASON: The solver restates a condition directly from the problem (reading half of the remaining pages) without performing any operation on it.
STEP 5: So, half of 84 is 42. Therefore, she should read 42 pages tomorrow. | APPLY | BLOOM_REASON: The solver applies a procedure by calculating half of the remaining pages to determine the required number of pages to read.
STEP 6: Let me double-check to make sure I didn't make a mistake. Total pages: 120. Yesterday: 12. Today: 24. Total read: 36. Remaining: 84. Half of 84 is 42. Yep, that seems right. | EVALUATE | BLOOM_REASON: The solver checks the correctness of the solution by verifying intermediate results and confirming that the final answer is consistent with the problem conditions.

#Example 2
Question:
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Reasoning trace:
Okay, let's tackle this problem step by step. So, Michael starts with 58 golf balls. On Tuesday, he lost 23. Then on Wednesday, he lost 2 more. I need to find out how many he has left by the end of Wednesday. First, let's figure out how many he had after Tuesday. He started with 58 and lost 23. So, subtract 23 from 58. Let me do that calculation. 58 minus 20 is 38, and then minus 3 more would be 35. Wait, 58 - 23. Let me check again: 58 minus 20 is 38, then subtract 3 more (since 23 is 20+3), so 38 - 3 = 35. Yeah, that's right. So after Tuesday, he has 35 golf balls left.Then on Wednesday, he lost 2 more. So now, take the 35 and subtract 2. 35 - 2 = 33. So, after Wednesday, he should have 33 golf balls. Let me double-check to make sure I didn't make a mistake. Starting with 58, lose 23: 58 - 23. Let's do it another way. 23 + 35 = 58, so yes, 58 -23 is 35. Then 35 -2 is 33. That seems right. So the answer should be 33.

Correct segmentation:
STEP 1: Okay, let's tackle this problem step by step. So, Michael starts with 58 golf balls. On Tuesday, he lost 23. Then on Wednesday, he lost 2 more. | REMEMBER | BLOOM_REASON: The solver restates facts given in the problem without any transformation or interpretation.
STEP 2: I need to find out how many he has left by the end of Wednesday. | UNDERSTAND | BLOOM_REASON: The solver interprets the goal of the problem by expressing what needs to be determined in their own words.
STEP 3: First, let's figure out how many he had after Tuesday. | ANALYZE | BLOOM_REASON: The solver breaks the problem into parts by identifying an intermediate quantity (the number of golf balls after Tuesday) needed to reach the final answer.
STEP 4: He started with 58 and lost 23. So, subtract 23 from 58. Let me do that calculation. 58 minus 20 is 38, and then minus 3 more would be 35. | APPLY | BLOOM_REASON: The solver executes arithmetic procedures by performing subtraction to compute the number of golf balls remaining after Tuesday.
STEP 5: Wait, 58 - 23. Let me check again: 58 minus 20 is 38, then subtract 3 more (since 23 is 20+3), so 38 - 3 = 35. Yeah, that's right. So after Tuesday, he has 35 golf balls left. | EVALUATE | BLOOM_REASON: The solver checks the correctness of the calculation by recomputing the subtraction and confirming that the result is accurate.
STEP 6: Then on Wednesday, he lost 2 more. So now, take the 35 and subtract 2. 35 - 2 = 33. So, after Wednesday, he should have 33 golf balls. | APPLY | BLOOM_REASON: The solver executes arithmetic procedures by subtracting the additional loss from the intermediate result to determine the number of golf balls remaining after Wednesday.
STEP 7: Let me double-check to make sure I didn't make a mistake. Starting with 58, lose 23: 58 - 23. Let's do it another way. 23 + 35 = 58, so yes, 58 -23 is 35. Then 35 -2 is 33. That seems right. So the answer should be 33. | EVALUATE | BLOOM_REASON: The solver verifies the correctness of the solution by rechecking calculations using an alternative method and confirming that the final answer is consistent.

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
        description="Annotate CoT steps with Bloom's taxonomy labels "
                    "(judge-segmented, Bloom-only version, vLLM)"
    )
    parser.add_argument("--judge_model_path", type=str, required=True,
                        help="Path to judge model, e.g. Llama-3.3-70B-Instruct")
    parser.add_argument("--judge_name", type=str, default=None,
                        help="Short name for the judge model. Auto-detected if omitted.")
    parser.add_argument("--input_dir", type=str, default="outputs",
                        help="Directory with cot_*.json output files")
    parser.add_argument("--pattern", type=str, default="*.json",
                        help="Glob pattern to select input files")
    parser.add_argument("--output_dir", type=str, default="outputs_bloom_annotated",
                        help="Directory to save annotated JSONs")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per file, for quick testing")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Max tokens for judge response")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature, matching OpenRouter default in your script")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter; set to 1.0 to match OpenRouter-like default")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Checkpoint every N completed annotation tasks")
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


def parse_judge_response(response_text):
    """Parse the judge's response into per-step Bloom labels.

    Accepts both short and long label variants and normalizes to:
      REMEMBER, UNDERSTAND, APPLY, ANALYZE, EVALUATE, CREATE, NONE
    """
    raw_to_canonical = {
        "REMEMBER": "REMEMBER",
        "REMEMBERING": "REMEMBER",
        "UNDERSTAND": "UNDERSTAND",
        "UNDERSTANDING": "UNDERSTAND",
        "APPLY": "APPLY",
        "APPLYING": "APPLY",
        "ANALYZE": "ANALYZE",
        "ANALYZING": "ANALYZE",
        "EVALUATE": "EVALUATE",
        "EVALUATING": "EVALUATE",
        "CREATE": "CREATE",
        "CREATING": "CREATE",
        "NONE": "NONE",
    }

    level_pattern = "|".join(
        sorted(raw_to_canonical.keys(), key=len, reverse=True)
    )

    step_numbers = [
        int(n) for n in re.findall(r"STEP\s+(\d+)", response_text, flags=re.IGNORECASE)
    ]
    if not step_numbers:
        return []

    max_step = max(step_numbers)
    step_labels = []

    for i in range(1, max_step + 1):
        pattern = (
            rf"STEP\s+{i}(?!\d)\s*:\s*(.*?)"
            rf"\s*\|\s*\**({level_pattern})\**"
            rf"\s*\|\s*\**BLOOM[_ ]REASON\**\s*:\s*(.*?)"
            rf"(?=\n\s*STEP\s+\d+\s*:|\Z)"
        )
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            step_text = match.group(1).strip().strip('"').strip()
            raw_level = match.group(2).strip().upper()
            bloom_reason = match.group(3).strip()
            step_labels.append({
                "step": step_text,
                "bloom_level": raw_to_canonical.get(raw_level, "NONE"),
                "bloom_reasoning": bloom_reason,
            })
        else:
            step_labels.append({
                "step": "",
                "bloom_level": "PARSE_ERROR",
                "bloom_reasoning": "",
            })

    return step_labels


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
    print("  Bloom level distribution:")
    for lv in BLOOM_LEVELS + ["PARSE_ERROR"]:
        c = bloom_counts.get(lv, 0)
        if c > 0:
            print(f"    {lv:15s}: {c:>5} ({c/total:.1%})")


def save_checkpoint(output_file, data, results, judge_name):
    out_data = {**data, "results": results}
    out_data.setdefault("summary", {})["bloom_judge"] = judge_name
    out_data["summary"]["bloom_judge_version"] = BLOOM_JUDGE_VERSION
    with open(output_file, "w") as f:
        json.dump(out_data, f, indent=2)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(args.pattern))
    if not input_files:
        print(f"No files matching {args.pattern} found in {input_dir}")
        return

    print(f"Found {len(input_files)} input file(s)")

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
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Judge model loaded.")
    print(f"Sampling params: max_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}")

    for input_file in input_files:
        print(f"\n{'=' * 60}")
        print(f"Processing {input_file.name}")
        print(f"{'=' * 60}")

        with open(input_file, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        file_model = summary.get("model", "unknown")
        file_dataset = summary.get("dataset", "unknown")
        results = data.get("results", [])

        if args.max_samples is not None:
            results = results[:args.max_samples]
            print(f"  Limited to {args.max_samples} samples (test mode)")

        print(f"  {len(results)} samples from {file_model}/{file_dataset}")

        output_file = output_dir / input_file.name

        start_idx = 0
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [warn] Corrupt/empty checkpoint {output_file.name}, starting fresh")
                existing_data = {"results": []}

            existing_results = existing_data.get("results", [])
            for i, r in enumerate(existing_results):
                has_output = "bloom_labels" in r or not r.get("reasoning", "").strip()
                has_thinking = "thinking_bloom_labels" in r or not r.get("thinking_trace", "").strip()
                if has_output and has_thinking:
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

        to_annotate = []
        for idx in range(start_idx, len(results)):
            r = results[idx]

            reasoning = r.get("reasoning", "")
            if reasoning.strip() and "bloom_labels" not in r:
                to_annotate.append((idx, reasoning, "bloom_labels"))
            elif "bloom_labels" not in r:
                results[idx]["bloom_labels"] = []

            thinking = r.get("thinking_trace", "")
            if thinking.strip() and "thinking_bloom_labels" not in r:
                to_annotate.append((idx, thinking, "thinking_bloom_labels"))
            elif "thinking_trace" in r and not thinking.strip() and "thinking_bloom_labels" not in r:
                results[idx]["thinking_bloom_labels"] = []

        print(f"  {len(to_annotate)} annotation tasks to process")

        prompts = []
        for idx, reasoning_text, label_key in to_annotate:
            r = results[idx]
            question = r["question"]
            prompts.append(build_chat_prompt(
                tokenizer,
                question,
                reasoning_text,
                judge_name=args.judge_name,
            ))

        all_labels = []
        completed_tasks = 0

        for batch_start in tqdm(range(0, len(prompts), args.batch_size), desc=input_file.stem):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_items = to_annotate[batch_start:batch_end]

            outputs = llm.generate(batch_prompts, sampling_params)

            for output, (idx, reasoning_text, label_key) in zip(outputs, batch_items):
                judge_response = output.outputs[0].text.strip()
                step_labels = parse_judge_response(judge_response)

                results[idx][label_key] = step_labels
                results[idx]["bloom_judge"] = args.judge_name
                results[idx]["bloom_judge_version"] = BLOOM_JUDGE_VERSION

                raw_key = (
                    "bloom_judge_raw"
                    if label_key == "bloom_labels"
                    else "thinking_bloom_judge_raw"
                )
                results[idx][raw_key] = judge_response
                all_labels.extend(step_labels)

                completed_tasks += 1

            if completed_tasks % args.save_every < len(batch_items) or batch_end == len(prompts):
                save_checkpoint(output_file, data, results, args.judge_name)

        save_checkpoint(output_file, data, results, args.judge_name)
        print_summary(f"{file_model} / {file_dataset}", all_labels)

    print("\nDone. Annotated files saved to", output_dir)


if __name__ == "__main__":
    main()
