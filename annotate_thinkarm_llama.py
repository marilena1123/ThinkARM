"""
Annotate ThinkARM using original repo's prompt with local Llama judge.

Simple wrapper around the original method/utils.py that uses local vLLM
instead of OpenAI API.

Usage:
    python annotate_thinkarm_llama.py \
        --response_model deepseekR1 \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --output_dir data/label
"""

import os
import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from method.utils import (
    process_response_to_sentences,
    guidebook_sentence_prompt
)
import re

parser = argparse.ArgumentParser()
parser.add_argument('--response_model', type=str, required=True, help='e.g., deepseekR1')
parser.add_argument('--judge_model_path', type=str, required=True, help='Path to Llama model')
parser.add_argument('--output_dir', type=str, default='output_thinkarm_llama', help='Output directory')
args = parser.parse_args()


def process_new_data_llama(new_data, sample_index, model_path, output_path):
    """Process data using local Llama judge (instead of OpenAI)."""

    # Initialize vLLM
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)

    new_instruction = new_data['Instruction']
    new_response = new_data['Response']

    new_sentence_list = process_response_to_sentences(new_response, apply_merging=True)

    general_sentence_instruction_prompt = """In this project, we aim to analyze the reasoning process of current large language models (LLMs) with advanced reasoning capabilities, i.e., Large Reasoning Models, LRMs, based on a modified version of Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. Given the model response you need to annotate the sentence-level behavior of the model response with the eight categories: Read, Analyze, Explore, Plan, Implement, Verify, Monitor, and Answer."""

    general_sentence_instruction_prompt += "\n\nThe [Guidebook] - [End of the Guidebook] section provides the detailed introduction and definition of each category."

    general_sentence_instruction_prompt += """\nThe [Math Problem] - [End of the Math Problem] section provides a math problem.\nThe [Overall Response] - [End of the Overall Response] section provides the overall response of the model to the math problem.\nThe [Previous Context] - [End of the Previous Context] section provides all the previous context of the response that has been annotated and their corresponding labels.\nThe [Input] - [End of the Input] section provides the sentences that need to be annotated.\nThe [Format] - [End of the Format] section provides the format of the output."""

    format_sentence_prompt = (
        "You should format the output in json format regarding the index, a short reasonale and the fine-grained class of the indexed sentence. "
        "The format is as follows:\n"
        "{\n"
        "  'sentences': [\n"
        "    {'index': 'The index of the sentence', 'reason': 'The short reason of the classification', 'category': 'The fine-grained class of the sentence'},\n"
        "    {'index': 'The index of the sentence', 'reason': 'The short reason of the classification', 'category': 'The fine-grained class of the sentence'},\n"
        "    ...\n"
        "  ]\n"
        "}"
        "You should strictly follow the index number of the sentence in the [Input] - [End of the Input] section."
    )

    batch_size = 20
    sen_list = []

    for idx, new_response_sentence in enumerate(new_sentence_list):
        if idx % batch_size == batch_size - 1 or idx == len(new_sentence_list) - 1:
            if idx == batch_size - 1 or len(new_sentence_list) < batch_size:
                new_input_context_prompt = "There is no previous sentences."
            else:
                new_input_context_list = new_sentence_list[: idx + 1 - batch_size]
                new_input_context_str = "\n".join([f"{item['sentence']}" for item in new_input_context_list])
                new_input_context_prompt = f"The previous sentences are:\n\n {new_input_context_str}"

            if len(new_sentence_list) < batch_size:
                new_input_list = new_sentence_list
            elif idx == len(new_sentence_list) - 1:
                remain = idx % batch_size + 1
                new_input_list = new_sentence_list[-remain:]
            else:
                new_input_list = new_sentence_list[idx-batch_size + 1: idx + 1]

            indexed_input_list = [f"[{idx+1}] {split['sentence']}" for idx, split in enumerate(new_input_list)]
            new_input_str = "\n".join(indexed_input_list)
            new_input_prompt = f"The following sentences which you need to classify:\n{new_input_str}"

            combined_prompt = f"{general_sentence_instruction_prompt}"
            combined_prompt += f"\n\n[Guidebook]\n{guidebook_sentence_prompt}\n[End of the Guidebook]"
            combined_prompt += f"\n\n[Math Problem]\n{new_instruction}\n[End of the Math Problem]\n\n[Previous Context]\n{new_input_context_prompt}\n[End of the Previous Context]\n\n[Input]\n{new_input_prompt}\n[End of the Input]\n\n[Format]\n{format_sentence_prompt}\n[End of the Format]\n\nNow, annotate the sentences in the [Input] - [End of the Input] section. Refer to the guidebook to make the decision. Strictly follow the index number of the sentence in the [Input] - [End of the Input] section for labeling. You should output the label for {len(new_input_list)} sentences."

            # Call local Llama
            outputs = llm.generate([combined_prompt], sampling_params)
            result = outputs[0].outputs[0].text.strip()

            # Parse JSON response
            try:
                json_match = re.search(r'\{[\s\S]*\}', result)
                response_json = json.loads(json_match.group())['sentences']
            except:
                response_json = []

            for item in response_json:
                group_index = int(item['index'].strip('[]'))
                item['sentence'] = new_input_list[int(group_index) - 1]['sentence']
                item['sentence-type'] = new_input_list[int(group_index) - 1]['type']

            response_json = [{
                'index': int(item['index'].strip('[]')) + (idx // batch_size) * batch_size,
                'sentence': item['sentence'],
                'sentence-type': item['sentence-type'],
                'sentence-category-reason': item['reason'],
                'sentence-category': item['category']
            } for item in response_json]

            sen_list.extend(response_json)

    # Save output
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/{sample_index + 1}.json", "w") as f:
        json.dump(sen_list, f, indent=2)

    return sen_list


def main():
    # Load raw data
    with open(f"data/raw/{args.response_model}.json", "r") as f:
        new_data = json.load(f)

    output_path = f"{args.output_dir}/{args.response_model}"

    print(f"Annotating {args.response_model} with Llama judge...")
    print(f"Output: {output_path}")

    for idx, data in enumerate(new_data):
        print(f"Processing {idx+1}/{len(new_data)}...")
        try:
            process_new_data_llama(data, idx, args.judge_model_path, output_path)
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue


if __name__ == '__main__':
    main()
