"""
Thinking trace generation script for ThinkARM.
Uses vLLM for fast batched inference on local models.

Adapted for offline HPC clusters with local models and datasets.

Usage:
    python generate.py \
        --model_path /path/to/model \
        --model_name qwen3_30b_thinking \
        --dataset_path /path/to/dataset.json \
        --dataset_name gsm8k \
        --output_dir ./outputs \
        --save_every 50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thinking traces using vLLM"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to locally saved HF model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Short model name for output filenames",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset JSON (local)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name for output filenames (e.g. gsm8k)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generation results",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save checkpoint every N examples",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index within the dataset",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index within the dataset (exclusive)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of prompts to send to vLLM at once",
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
    return parser.parse_args()


def is_thinking_model(model_name):
    """Whether this model produces <think>...</think> blocks."""
    name = model_name.lower()
    return "thinking" in name or "deepseek_r1" in name or "phi_4_reasoning" in name


def strip_thinking_tokens(text):
    """Remove <think>...</think> blocks from thinking model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned and cleaned != text:
        return cleaned
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def needs_chat_template(model_name):
    """Whether to wrap prompts in the tokenizer's chat template."""
    name = model_name.lower()
    return (
        is_thinking_model(model_name)
        or "instruct" in name
        or "sft" in name
        or "hybrid" in name
    )


def extract_thinking_trace(text):
    """Extract the thinking trace from thinking model output."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()
    return None


def split_cot_steps(reasoning_text):
    """Split reasoning into sentence-level steps."""
    sentences = re.split(r"(?<=[.!?])\s+", reasoning_text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_final_answer(text):
    """Extract the final answer from generated text."""
    # Look for "The answer is X" pattern
    pattern = r"[Tt]he answer is\s+(.+?)(?:\.|$)"
    match = re.search(pattern, text)
    if match:
        raw = match.group(1).strip().rstrip(".")
        # Try numeric extraction
        num_match = re.match(r"^[\$]?\s*(-?\d[\d,]*\.?\d*)", raw)
        if num_match:
            return num_match.group(1).replace(",", "")
        # Letter answer (A, B, C, D)
        letter_match = re.match(r"^([A-Z])\b", raw)
        if letter_match:
            return letter_match.group(1)
        return raw

    # Fallback: last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_ground_truth(answer_text):
    """Extract answer from ground truth."""
    s = str(answer_text).strip()

    # Multiple-choice letter
    if re.match(r"^[A-Z]$", s):
        return s

    # GSM8K format: #### <number>
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", s)
    if match:
        return match.group(1).replace(",", "")

    # Plain number
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if numbers:
        return numbers[-1].replace(",", "")

    return s.strip()


def is_answer_correct(predicted, ground_truth):
    """Numeric-tolerant string comparison."""
    if predicted is None or ground_truth is None:
        return False
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        return predicted.strip().lower() == ground_truth.strip().lower()


def process_outputs(outputs, examples, model_name, dataset_name, tokenizer=None):
    """Process vLLM outputs into result dicts."""
    batch_results = []
    for output, example in zip(outputs, examples):
        # Decode output
        if is_thinking_model(model_name) and tokenizer is not None:
            raw_output = tokenizer.decode(
                output.outputs[0].token_ids,
                skip_special_tokens=False,
            ).strip()
            raw_output = re.sub(r"<[\|｜][^>]*[\|｜]>", "", raw_output).strip()
        else:
            raw_output = output.outputs[0].text.strip()

        # Extract thinking trace
        thinking_trace = None
        if is_thinking_model(model_name):
            thinking_trace = extract_thinking_trace(raw_output)
            reasoning = strip_thinking_tokens(raw_output)
        else:
            reasoning = raw_output

        # Extract answers
        predicted = extract_final_answer(reasoning)
        ground_truth = extract_ground_truth(example.get("answer", ""))
        is_correct = is_answer_correct(predicted, ground_truth)

        result = {
            "question": example["question"],
            "reasoning": reasoning,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "ground_truth_raw": example.get("answer", ""),
            "correct": is_correct,
            "steps": split_cot_steps(reasoning),
        }

        # Add task if present
        if "task" in example:
            result["task"] = example["task"]

        # Handle thinking traces
        if thinking_trace is not None:
            result["thinking_trace"] = thinking_trace
            result["thinking_steps"] = split_cot_steps(thinking_trace)
            thinking_predicted = extract_final_answer(thinking_trace)
            result["thinking_predicted_answer"] = thinking_predicted
            result["thinking_correct"] = is_answer_correct(
                thinking_predicted, ground_truth
            )

        batch_results.append(result)

    return batch_results


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"gen_{args.model_name}_{args.dataset_name}.json"
    output_path = output_dir / output_filename

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    # Slice if requested
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset = dataset[args.start_idx : end_idx]
    print(f"Running generation on {len(dataset)} samples "
          f"({args.dataset_name}, {args.model_name})")

    # Load existing results if resuming
    results = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            results = existing
        elif isinstance(existing, dict) and "results" in existing:
            results = existing["results"]
        print(f"Loaded {len(results)} existing results, resuming...")
        dataset = dataset[len(results):]

    if not dataset:
        print("All samples already processed.")
        return

    # Initialize vLLM
    print(f"Loading model from {args.model_path} with vLLM...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )

    # Sampling params
    if is_thinking_model(args.model_name):
        if "phi_4_reasoning" in args.model_name.lower():
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                min_p=0.0,
                max_tokens=args.max_new_tokens,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                max_tokens=args.max_new_tokens,
            )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_tokens=args.max_new_tokens,
        )

    # Get tokenizer if needed
    vllm_tokenizer = None
    if needs_chat_template(args.model_name):
        vllm_tokenizer = llm.get_tokenizer()

    # Simple prompt template
    prompt_template = "Question: {question}\n\nAnswer:"

    # Build prompts
    raw_prompts = [prompt_template.format(question=ex["question"]) for ex in dataset]

    # Apply chat template if needed
    apply_kwargs = {"tokenize": False, "add_generation_prompt": True}
    force_think_prefix = "deepseek_r1" in args.model_name.lower()

    if vllm_tokenizer is not None:
        prompts = []
        for raw in raw_prompts:
            messages = [{"role": "user", "content": raw}]
            chat_prompt = vllm_tokenizer.apply_chat_template(
                messages, **apply_kwargs
            )
            if force_think_prefix:
                chat_prompt += "<think>\n"
            prompts.append(chat_prompt)
    else:
        prompts = raw_prompts

    # Process in batches
    correct = sum(1 for r in results if r.get("correct", False))
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_examples = dataset[batch_start:batch_end]

        outputs = llm.generate(batch_prompts, sampling_params)
        batch_results = process_outputs(
            outputs, batch_examples, args.model_name, args.dataset_name,
            tokenizer=vllm_tokenizer,
        )

        for result in batch_results:
            if result["correct"]:
                correct += 1
            results.append(result)

        # Checkpoint
        total_so_far = len(results)
        if total_so_far % args.save_every < batch_size or batch_end == len(prompts):
            accuracy = correct / total_so_far if total_so_far > 0 else 0.0
            print(f"  [{total_so_far}] accuracy so far: {accuracy:.4f}")
            checkpoint_data = {
                "summary": {
                    "_summary": True,
                    "_checkpoint": True,
                    "model": args.model_name,
                    "dataset": args.dataset_name,
                    "total": total_so_far,
                    "correct": correct,
                    "accuracy": accuracy,
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

    # Final save
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "_summary": True,
        "model": args.model_name,
        "dataset": args.dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    output_data = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone. Generated | {args.model_name} | {args.dataset_name}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
