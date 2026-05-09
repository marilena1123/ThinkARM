"""
Annotate ThinkARM thinking traces with ThinkARM episodes — using repository's exact prompt.

Extracts thinking traces from data/raw/*.json and annotates them with the original
ThinkARM episode categories using local Llama judge model.

Episode categories: Read, Analyze, Plan, Implement, Explore, Verify, Monitor, Answer

Output schema (per model):
  output_dir/{model_name}.json — list of annotated samples
    Each sample: list of { index, sentence, sentence_type, sentence_category, sentence_category_reason }

Usage:
    python annotate_thinkarm_thinking_episodes.py \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --input_dir data/raw \
        --output_dir output_thinkarm_thinking_episodes \
        --models deepseekR1 "DeepSeek-R1-Distill-Qwen-1.5B" "DeepSeek-R1-Distill-Qwen-7B" Phi4R
"""

import os
import json
import re
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import gc


# Load guidebook
with open("guidebook/sentence_guide.md", "r") as f:
    guidebook_sentence_prompt = f.read()


def split_response_into_paragraphs(response):
    return [p.strip() for p in response.split('\n\n')]


def split_paragraph_into_sentences(paragraph):
    splits = []
    current = ''
    i = 0
    in_math_block = False
    math_delimiter = None

    abbreviations = {
        'e.g.', 'i.e.', 'v.s.', 'cf.', 'et al.', 'ibid.', 'etc.', 'vs.', 'viz.',
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Rev.', 'St.', 'Jr.', 'Sr.',
        'Inc.', 'Ltd.', 'Corp.', 'Co.', 'LLC.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
        'U.S.', 'U.K.', 'U.S.A.', 'N.Y.', 'L.A.', 'D.C.', 'a.m.', 'p.m.',
        'No.', 'Vol.', 'pp.', 'Fig.', 'Eq.', 'Ref.', 'Sec.', 'Ch.', 'App.'
    }

    def is_abbreviation_context(text, position):
        for abbrev in abbreviations:
            abbrev_len = len(abbrev)
            start_pos = position - abbrev_len + 1
            if start_pos >= 0 and position + 1 <= len(text):
                potential_abbrev = text[start_pos:position + 1]
                if potential_abbrev.lower() == abbrev.lower():
                    if start_pos == 0 or not text[start_pos - 1].isalnum():
                        return True

        for abbrev in abbreviations:
            abbrev_len = len(abbrev)
            for start_offset in range(abbrev_len):
                start_pos = position - start_offset
                end_pos = start_pos + abbrev_len
                if (start_pos >= 0 and end_pos <= len(text) and
                    start_pos <= position < end_pos):
                    potential_abbrev = text[start_pos:end_pos]
                    if potential_abbrev.lower() == abbrev.lower():
                        if start_pos == 0 or not text[start_pos - 1].isalnum():
                            return True
        return False

    while i < len(paragraph):
        char = paragraph[i]
        current += char

        if char == '$':
            if not in_math_block:
                if i + 1 < len(paragraph) and paragraph[i + 1] == '$':
                    math_delimiter = '$$'
                    current += '$'
                    i += 1
                else:
                    math_delimiter = '$'
                in_math_block = True
            else:
                if math_delimiter == '$$' and i + 1 < len(paragraph) and paragraph[i + 1] == '$':
                    current += '$'
                    i += 1
                    in_math_block = False
                    math_delimiter = None
                elif math_delimiter == '$':
                    in_math_block = False
                    math_delimiter = None

        elif not in_math_block:
            if char == '.' and i + 2 < len(paragraph) and paragraph[i+1:i+3] == '..':
                current += paragraph[i+1:i+3]
                i += 2

                context_before = current[-20:] if len(current) >= 20 else current
                context_after = paragraph[i+1:i+21] if i+1 < len(paragraph) else ""

                math_indicators = ['+', '-', '*', '/', '=', '(', ')', '[', ']', 'g(', 'f(', 'h(', 'times', 'integer', 'induction']

                is_math_context = any(indicator in context_before.lower() or indicator in context_after.lower()
                                    for indicator in math_indicators)

                if not is_math_context:
                    splits.append(current.strip())
                    current = ''
            elif char in '.?!':
                if char == '.' and is_abbreviation_context(paragraph, i):
                    pass
                elif char == '.' and i > 0 and i < len(paragraph)-1:
                    prev_char = paragraph[i-1]
                    next_char = paragraph[i+1]
                    if prev_char.isdigit() and next_char.isdigit():
                        pass
                    elif i == 1 and prev_char.isdigit():
                        pass
                    else:
                        splits.append(current.strip())
                        current = ''
                else:
                    splits.append(current.strip())
                    current = ''

        i += 1

    if current:
        splits.append(current.strip())
    return splits


def is_valid_sentence(sentence):
    if not sentence or not sentence.strip():
        return False

    cleaned = sentence.strip()

    if all(c == '-' for c in cleaned):
        return False

    alphanumeric_chars = ''.join(c for c in cleaned if c.isalnum())
    return bool(alphanumeric_chars)


def process_section(section_text, section_type):
    paragraphs = split_response_into_paragraphs(section_text)
    sentences = []

    for paragraph in paragraphs:
        paragraph_sentences = split_paragraph_into_sentences(paragraph)
        for sentence in paragraph_sentences:
            if is_valid_sentence(sentence):
                sentences.append({
                    'sentence': sentence,
                    'type': section_type
                })
    return sentences


def merge_colon_and_equals_sentences(sentences):
    merged = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i].copy()
        current_sentence['sentence'] = current_sentence['sentence'].replace('<think>', '').strip()

        while (current_sentence['sentence'].rstrip().endswith(':') and
               i + 1 < len(sentences)):
            next_sentence = sentences[i + 1].copy()
            next_sentence['sentence'] = next_sentence['sentence'].replace('<think>', '').strip()

            if current_sentence['type'] == next_sentence['type']:
                current_sentence['sentence'] = current_sentence['sentence'] + ' ' + next_sentence['sentence']
                i += 1
            else:
                break

        merged.append(current_sentence)
        i += 1

    final_merged = []
    for i, sentence in enumerate(merged):
        sentence_copy = sentence.copy()
        sentence_copy['sentence'] = sentence_copy['sentence'].replace('<think>', '').strip()

        if (sentence_copy['sentence'].lstrip().startswith('=') and
            len(final_merged) > 0 and
            final_merged[-1]['type'] == sentence_copy['type']):
            final_merged[-1]['sentence'] = final_merged[-1]['sentence'] + ' ' + sentence_copy['sentence']
        else:
            final_merged.append(sentence_copy)

    return final_merged


def process_response_to_sentences(response, apply_merging=True):
    """Process a response into structured sentences."""
    if '</think>' in response:
        parts = response.split('</think>', 1)
        thinking_part = parts[0].strip()

        thinking_sentences = process_section(thinking_part, 'think')
        all_sentences = thinking_sentences
    else:
        all_sentences = process_section(response, 'answer')

    if apply_merging:
        processed_sentences = merge_colon_and_equals_sentences(all_sentences)
    else:
        processed_sentences = all_sentences

    result = []
    for i, sentence_data in enumerate(processed_sentences):
        result.append({
            'id': str(i),
            'sentence': sentence_data['sentence'],
            'type': sentence_data['type']
        })

    return result


def build_annotation_prompt(instruction, response_text, sentence_list):
    """Build annotation prompt using simplified format."""

    format_instruction = """You are a data annotation expert. Classify each sentence according to ThinkARM's 8 episode categories:
- Read: Extract/restate given information and problem goal
- Analyze: Logical inference and relationship deduction
- Plan: State intended next step or strategy
- Implement: Execute procedures and calculations
- Explore: Tentative reasoning and possibilities
- Verify: Check correctness or validity
- Monitor: Meta-commentary (e.g., "Let me think")
- Answer: Deliver the final answer

Output ONLY valid JSON with no other text. Keep reasons VERY SHORT (1-2 words):
{
  "sentences": [
    {"index": "1", "reason": "introduces problem", "category": "Read"},
    {"index": "2", "reason": "states condition", "category": "Read"}
  ]
}"""

    indexed_input_list = [f"[{idx+1}] {split['sentence']}" for idx, split in enumerate(sentence_list)]
    new_input_str = "\n".join(indexed_input_list)

    prompt = f"""Question: {instruction}

Reasoning to annotate:
{response_text}

Sentences to classify:
{new_input_str}

{format_instruction}

Classify {len(sentence_list)} sentences. Output ONLY the JSON object, nothing else. Keep reasons SHORT."""

    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate ThinkARM thinking traces with ThinkARM episode categories"
    )
    parser.add_argument("--judge_model_path", type=str, required=True,
                        help="Path to judge model (e.g. Llama-3.3-70B-Instruct)")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Directory with raw JSON files")
    parser.add_argument("--models", type=str, nargs='+',
                        default=["deepseekR1", "DeepSeek-R1-Distill-Qwen-1.5B",
                                "DeepSeek-R1-Distill-Qwen-7B", "Phi4R"],
                        help="Model names to process")
    parser.add_argument("--output_dir", type=str, default="output_thinkarm_thinking_episodes",
                        help="Directory to save annotated JSONs")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per file (for quick testing)")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Max tokens for judge response")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Checkpoint every N samples")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of prompts to send to vLLM at once")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    args = parser.parse_args()
    return args


def annotate_thinking(llm, sampling_params, instruction, thinking_text, sample_idx):
    """Annotate a single thinking trace with ThinkARM episodes."""
    sentence_list = process_response_to_sentences(thinking_text, apply_merging=True)

    if not sentence_list:
        return None

    prompt = build_annotation_prompt(instruction, thinking_text, sentence_list)

    try:
        outputs = llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text.strip()

        # Parse JSON response - try multiple approaches
        response_json = None

        # Try to extract JSON from markdown code blocks first
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', result)
        if code_block_match:
            json_str = code_block_match.group(1)
            try:
                response_json = json.loads(json_str)['sentences']
            except (json.JSONDecodeError, KeyError):
                # Try fixing common issues: unescaped newlines in strings
                json_str = re.sub(r':\s*"([^"]*)\n([^"]*)"', r': "\1 \2"', json_str)
                try:
                    response_json = json.loads(json_str)['sentences']
                except (json.JSONDecodeError, KeyError):
                    pass

        # If that fails, try to find JSON object in the response
        if not response_json:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                json_str = json_match.group()
                try:
                    response_json = json.loads(json_str)['sentences']
                except json.JSONDecodeError as e:
                    # Try fixing common issues: unescaped newlines in strings
                    json_str = re.sub(r':\s*"([^"]*)\n([^"]*)"', r': "\1 \2"', json_str)
                    try:
                        response_json = json.loads(json_str)['sentences']
                    except json.JSONDecodeError:
                        print(f"  [warn] Sample {sample_idx}: JSON parse error - {str(e)[:50]}")
                        print(f"  [warn] First 200 chars of response: {result[:200]}")
                        return None

        if not response_json:
            print(f"  [warn] Sample {sample_idx}: No valid JSON found in response")
            return None

        # Add sentence text and type to annotations
        for item in response_json:
            try:
                group_index = int(item['index'].strip('[]')) - 1
                if group_index < len(sentence_list):
                    item['sentence'] = sentence_list[group_index]['sentence']
                    item['sentence_type'] = sentence_list[group_index]['type']
            except (KeyError, ValueError, TypeError):
                pass

        # Reformat to match ThinkARM output format
        formatted = []
        for i, item in enumerate(response_json):
            formatted.append({
                'index': i + 1,
                'sentence': item.get('sentence', ''),
                'sentence_type': item.get('sentence_type', 'think'),
                'sentence_category': item.get('category', ''),
                'sentence_category_reason': item.get('reason', ''),
            })

        return formatted

    except Exception as e:
        print(f"  [error] Sample {sample_idx}: {str(e)[:100]}")
        return None


def extract_thinking_trace(response_text):
    """Extract thinking trace from Response field."""
    if not response_text:
        return ""

    if "<think>" in response_text:
        match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return response_text.strip()


def main():
    try:
        args = parse_args()

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load vLLM once
        print(f"Loading model: {args.judge_model_path}")
        llm = LLM(
            model=args.judge_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            enforce_eager=True,
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        print("Model loaded successfully")

        # Process each model
        for model_name in args.models:
            input_file = input_dir / f"{model_name}.json"

            if not input_file.exists():
                print(f"\n[warn] File not found: {input_file}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {model_name}")
            print(f"{'='*60}")

            # Load data
            with open(input_file, "r") as f:
                data = json.load(f)

            results = data if isinstance(data, list) else data.get("results", [])

            if args.max_samples is not None:
                results = results[:args.max_samples]
                print(f"Limited to {args.max_samples} samples (test mode)")

            print(f"Found {len(results)} samples")

            # Collect all annotations for this model
            all_annotations = []
            processed = 0
            skipped = 0

            # Process each sample
            for idx, result in enumerate(tqdm(results, desc=model_name)):
                instruction = result.get("Instruction", result.get("question", ""))
                response = result.get("Response", "")
                thinking = extract_thinking_trace(response)

                if not thinking.strip():
                    skipped += 1
                    continue

                annotations = annotate_thinking(llm, sampling_params, instruction, thinking, idx)
                if annotations:
                    all_annotations.append(annotations)
                    processed += 1
                else:
                    skipped += 1

            # Save aggregated results to single JSON file per model
            output_file = output_dir / f"{model_name}.json"
            with open(output_file, "w") as f:
                json.dump(all_annotations, f, indent=2)

            print(f"✅ Completed: {model_name}")
            print(f"   Processed: {processed}, Skipped: {skipped}/{len(results)}")
            print(f"   Output: {output_file}")

        # Cleanup GPU memory
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        print("\n✅ All models annotated successfully!")

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        raise


if __name__ == "__main__":
    main()
