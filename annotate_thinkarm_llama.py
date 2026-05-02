"""
Annotate ThinkARM using original repo's prompt with local Llama judge.

Uses the exact pipeline from the original ThinkARM repository,
but runs with local vLLM + Llama instead of OpenAI API.

Usage:
    python annotate_thinkarm_llama.py \
        --response_model deepseekR1 \
        --judge_model_path /path/to/Llama-3.3-70B-Instruct \
        --output_dir output_thinkarm_llama
"""

import os
import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
import re


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
        answer_part = parts[1].strip()

        thinking_sentences = process_section(thinking_part, 'think')
        answer_sentences = process_section(answer_part, 'answer')

        all_sentences = thinking_sentences + answer_sentences
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
