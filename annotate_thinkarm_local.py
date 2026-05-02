"""
Annotate ThinkARM thinking traces using ORIGINAL repository prompt with local Llama judge.

Uses the exact prompt and guidebook from the original ThinkARM repository,
but runs with local vLLM + Llama-3.3-70B instead of OpenAI API.

Usage:
    python annotate_thinkarm_local.py \
        --input_path outputs_thinkarm_for_judge/bloom_input_*.json \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --output_dir outputs_thinkarm_judge_annotated \
        --batch_size 8
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re
from glob import glob

from vllm import LLM, SamplingParams
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate with original ThinkARM prompt using local judge"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="bloom_input_*.json",
        help="File pattern to match",
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        required=True,
        help="Path to Llama-3.3-70B-Instruct model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_thinkarm_judge_annotated",
        help="Directory to save annotated results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for vLLM",
    )
    return parser.parse_args()


# Load original guidebook
with open("guidebook/sentence_guide.md", "r") as f:
    GUIDEBOOK = f.read()


def build_annotation_prompt(
    instruction: str,
    response: str,
    sentence_list: List[Dict[str, Any]]
) -> str:
    """Build annotation prompt using ORIGINAL format from repository."""

    # Original general instruction
    general_instruction = """In this project, we aim to analyze the reasoning process of current large language models (LLMs) with advanced reasoning capabilities, i.e., Large Reasoning Models, LRMs, based on a modified version of Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. Given the model response you need to annotate the sentence-level behavior of the model response with the eight categories: Read, Analyze, Explore, Plan, Implement, Verify, Monitor, and Answer.

The [Guidebook] - [End of the Guidebook] section provides the detailed introduction and definition of each category.

The [Math Problem] - [End of the Math Problem] section provides a math problem.
The [Response] - [End of the Response] section provides the response of the model to the math problem.
The [Sentences] - [End of the Sentences] section provides the sentences that need to be annotated.
The [Format] - [End of the Format] section provides the format of the output."""

    # Format instruction
    format_instruction = """You should format the output in json format regarding the index, sentence, a short reason and the fine-grained class of the indexed sentence.
The format is as follows:
{
  "sentences": [
    {"index": 1, "sentence": "sentence text", "reason": "The short reason of the classification", "category": "The fine-grained class of the sentence"},
    {"index": 2, "sentence": "sentence text", "reason": "The short reason of the classification", "category": "The fine-grained class of the sentence"},
    ...
  ]
}
You should strictly follow the index number of the sentence in the [Sentences] - [End of the Sentences] section."""

    # Format input sentences
    indexed_sentences = "\n".join(
        [f"[{i+1}] {s['sentence']}" for i, s in enumerate(sentence_list)]
    )

    # Build combined prompt
    prompt = f"""{general_instruction}

[Guidebook]
{GUIDEBOOK}
[End of the Guidebook]

[Math Problem]
{instruction}
[End of the Math Problem]

[Response]
{response}
[End of the Response]

[Sentences]
{indexed_sentences}
[End of the Sentences]

[Format]
{format_instruction}
[End of the Format]

Now, annotate the sentences in the [Sentences] - [End of the Sentences] section. Refer to the guidebook to make the decision. Strictly follow the index number of the sentence in the [Sentences] - [End of the Sentences] section for labeling. You should output the label for {len(sentence_list)} sentences."""

    return prompt


def parse_annotation_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse JSON response from judge model."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            response_json = json.loads(json_match.group())
            return response_json.get("sentences", [])
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def annotate_file(
    llm: LLM,
    input_file: str,
    output_dir: str,
    sampling_params: SamplingParams,
) -> None:
    """Annotate a single input file."""

    with open(input_file, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    for result in tqdm(results, desc=f"Annotating {Path(input_file).name}"):
        question = result.get("question", "")
        reasoning = result.get("reasoning", "")
        thinkarm_sentences = result.get("thinkarm_original_sentences", [])

        if not thinkarm_sentences:
            continue

        # Extract sentence text from ThinkARM annotations
        sentence_list = [
            {"sentence": s.get("sentence", "")}
            for s in thinkarm_sentences
        ]

        # Build prompt using ORIGINAL format
        prompt = build_annotation_prompt(question, reasoning, sentence_list)

        # Prepare message
        messages = [
            {"role": "system", "content": "You are an expert in analyzing reasoning processes using Schoenfeld's Episode Theory framework."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        chat_prompt = llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate annotation
        outputs = llm.generate([chat_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        # Parse response
        annotations = parse_annotation_response(response_text)

        # Merge with original ThinkARM annotations
        for i, annotation in enumerate(annotations):
            if i < len(thinkarm_sentences):
                thinkarm_sentences[i]["judge_category"] = annotation.get("category")
                thinkarm_sentences[i]["judge_reason"] = annotation.get("reason")
                thinkarm_sentences[i]["judge_raw"] = response_text

        # Add judge annotations to result
        result["thinkarm_judge_labels"] = annotations
        result["thinkarm_judge_version"] = "original_prompt_vllm"
        result["thinkarm_judge_model"] = "llama-3.3-70b-instruct"

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / Path(input_file).name

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {output_file}")


def main():
    args = parse_args()

    # Initialize vLLM
    print(f"Loading model: {args.judge_model_path}")
    llm = LLM(
        model=args.judge_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        top_p=1.0
    )

    # Find input files
    input_dir = Path(args.input_dir)
    input_files = sorted(input_dir.glob(args.pattern))

    if not input_files:
        print(f"No files found matching {args.input_dir}/{args.pattern}")
        return

    print(f"Found {len(input_files)} files to annotate")

    # Annotate each file
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        annotate_file(llm, str(input_file), args.output_dir, sampling_params)

    print(f"\n✅ All files annotated and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
