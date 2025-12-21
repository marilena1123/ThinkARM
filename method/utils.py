import os
from google import genai
import json
from tqdm import tqdm
from transformers import AutoTokenizer

api_key = os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)

with open("guidebook/sentence_guide.md", "r") as f:
        guidebook_sentence_prompt = f.read()


def process_new_data(client, new_data, sample_index, guidebook=True, model="gpt-4.1", output_path='result', answer_key='Response'):


    new_instruction = new_data['Instruction']
    new_instruction_prompt = f"{new_instruction}"
    new_response = new_data[answer_key]

    new_sentence_list = process_response_to_sentences(new_response, apply_merging=True)
    
    general_sentence_instruction_prompt = """In this project, we aim to analyze the reasoning process of current large language models (LLMs) with advanced reasoning capabilities, i.e., Large Reasoning Models, LRMs, based on a modified version of Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. Given the model response you need to annotate the sentence-level behavior of the model response with the eight categories: Read, Analyze, Explore, Plan, Implement, Verify, Monitor, and Answer."""

    if guidebook:
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
    for idx, new_response_sentence in enumerate(tqdm(new_sentence_list)):
        if idx % batch_size == batch_size - 1 or idx == len(new_sentence_list) - 1:
            if idx == batch_size - 1 or len(new_sentence_list) < batch_size:
                new_input_context = ""
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
                # print(len(new_input_list))
            else:
                new_input_list = new_sentence_list[idx-batch_size + 1: idx + 1]
            indexed_input_list = [f"[{idx+1}] {split['sentence']}" for idx, split in enumerate(new_input_list)]
            new_input_str = "\n".join(indexed_input_list)
            new_input_prompt = f"The following sentences which you need to classify:\n{new_input_str}"

            combined_prompt = f"{general_sentence_instruction_prompt}"

            if guidebook:
                combined_prompt += f"\n\n[Guidebook]\n{guidebook_sentence_prompt}\n[End of the Guidebook]"

            combined_prompt += f"\n\n[Math Problem]\n{new_instruction_prompt}\n[End of the Math Problem]\n\n[Previous Context]\n{new_input_context_prompt}\n[End of the Previous Context]\n\n[Input]\n{new_input_prompt}\n[End of the Input]\n\n[Format]\n{format_sentence_prompt}\n[End of the Format]\n\nNow, annotate the sentences in the [Input] - [End of the Input] section. Refer to the guidebook to make the decision. Strictly follow the index number of the sentence in the [Input] - [End of the Input] section for labeling. You should output the label for {len(new_input_list)} sentences."

            messages = [
                {"role": "user", "content": combined_prompt},
            ]
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if 'gpt' in model or 'deepseek' in model:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            response_format={"type": "json_object"}
                        )
                        result = response.choices[0].message.content
                    elif 'gemini' in model:
                        response = google_client.models.generate_content(
                            model=model,
                            contents=combined_prompt,
                            config={
                                "response_mime_type": "application/json",
                                "response_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "sentences": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "index": {"type": "string"},
                                                    "reason": {"type": "string"},
                                                    "category": {"type": "string"}
                                                },
                                                "required": ["index", "reason", "category"],
                                                "propertyOrdering": ["index", "reason", "category"]
                                            }
                                        }
                                    },
                                    "required": ["sentences"],
                                    "propertyOrdering": ["sentences"]
                                }
                            }
                        )
                        result = response.text
                    response_json = json.loads(result)['sentences']
                    assert len(response_json) == len(new_input_list), f"Number of sentences in the response ({len(response_json)}) does not match the number of sentences in the input ({len(new_input_list)})"
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Attempt {retry_count} failed, retrying... Error: {str(e)}")
                    if retry_count == max_retries:
                        print(f"Failed to get response after {max_retries} retries. Using empty response.")
                        response = {'sentences': [{'index': '', 'sentence-category-reason': '', 'sentence-category': ''}]}

            # print(result)

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

    target_path = f"{output_path}"
    # for idx, item in enumerate(sen_list):
    #     item['index'] = idx + 1

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(f"{target_path}/{sample_index + 1}.json", "w") as f:
        json.dump(sen_list, f, indent=2)

    return sen_list


def split_response_into_paragraphs(response):
    return [p.strip() for p in response.split('\n\n')]

def split_paragraph_into_sentences(paragraph):
    splits = []
    current = ''
    i = 0
    in_math_block = False
    math_delimiter = None  # Track whether we're in $ or $$ block
    
    # Common abbreviations that shouldn't trigger sentence splits
    abbreviations = {
        'e.g.', 'i.e.', 'v.s.', 'cf.', 'et al.', 'ibid.', 'etc.', 'vs.', 'viz.',
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Rev.', 'St.', 'Jr.', 'Sr.',
        'Inc.', 'Ltd.', 'Corp.', 'Co.', 'LLC.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
        'U.S.', 'U.K.', 'U.S.A.', 'N.Y.', 'L.A.', 'D.C.', 'a.m.', 'p.m.',
        'No.', 'Vol.', 'pp.', 'Fig.', 'Eq.', 'Ref.', 'Sec.', 'Ch.', 'App.'
    }
    
    def is_abbreviation_context(text, position):
        """Check if the period at position is part of a known abbreviation"""
        # Look for abbreviations that end at this position
        for abbrev in abbreviations:
            abbrev_len = len(abbrev)
            start_pos = position - abbrev_len + 1
            if start_pos >= 0 and position + 1 <= len(text):
                # Extract the potential abbreviation from the text
                potential_abbrev = text[start_pos:position + 1]
                if potential_abbrev.lower() == abbrev.lower():
                    # Additional check: make sure this is a word boundary
                    # Check if there's a space, punctuation, or start of text before the abbreviation
                    if start_pos == 0 or not text[start_pos - 1].isalnum():
                        return True
        
        # Also check if this period is part of an abbreviation that continues after this position
        # Look for abbreviations that contain this position
        for abbrev in abbreviations:
            abbrev_len = len(abbrev)
            # Check all possible starting positions for this abbreviation
            for start_offset in range(abbrev_len):
                start_pos = position - start_offset
                end_pos = start_pos + abbrev_len
                if (start_pos >= 0 and end_pos <= len(text) and 
                    start_pos <= position < end_pos):
                    potential_abbrev = text[start_pos:end_pos]
                    if potential_abbrev.lower() == abbrev.lower():
                        # Check word boundary
                        if start_pos == 0 or not text[start_pos - 1].isalnum():
                            return True
        return False
    
    while i < len(paragraph):
        char = paragraph[i]
        current += char
        
        # Check for math delimiters
        if char == '$':
            if not in_math_block:
                # Check if it's $$ (display math) or $ (inline math)
                if i + 1 < len(paragraph) and paragraph[i + 1] == '$':
                    math_delimiter = '$$'
                    current += '$'
                    i += 1  # Skip the second $
                else:
                    math_delimiter = '$'
                in_math_block = True
            else:
                # We're already in a math block, check if this closes it
                if math_delimiter == '$$' and i + 1 < len(paragraph) and paragraph[i + 1] == '$':
                    current += '$'
                    i += 1  # Skip the second $
                    in_math_block = False
                    math_delimiter = None
                elif math_delimiter == '$':
                    in_math_block = False
                    math_delimiter = None
        
        # Only process sentence endings if we're not in a math block
        elif not in_math_block:
            # Check for ellipsis (...)
            if char == '.' and i + 2 < len(paragraph) and paragraph[i+1:i+3] == '..':
                # Add the remaining two dots to complete the ellipsis
                current += paragraph[i+1:i+3]
                i += 2  # Skip the next two dots
                
                # Check if this ellipsis is in a mathematical context
                # Look for mathematical indicators before or after
                context_before = current[-20:] if len(current) >= 20 else current
                context_after = paragraph[i+1:i+21] if i+1 < len(paragraph) else ""
                
                # Mathematical context indicators
                math_indicators = ['+', '-', '*', '/', '=', '(', ')', '[', ']', 'g(', 'f(', 'h(', 'times', 'integer', 'induction']
                
                is_math_context = any(indicator in context_before.lower() or indicator in context_after.lower() 
                                    for indicator in math_indicators)
                
                # Only split if it's not in a mathematical context
                if not is_math_context:
                    splits.append(current.strip())
                    current = ''
            elif char in '.?!':
                # Check if dot is part of an abbreviation
                if char == '.' and is_abbreviation_context(paragraph, i):
                    # This period is part of an abbreviation, don't split here
                    pass  # Continue with the loop, don't split
                elif char == '.' and i > 0 and i < len(paragraph)-1:
                    # Check if dot is between numbers (decimal point)
                    prev_char = paragraph[i-1]
                    next_char = paragraph[i+1]
                    if prev_char.isdigit() and next_char.isdigit():
                        pass  # Continue with the loop, don't split
                    elif i == 1 and prev_char.isdigit():
                        pass  # Continue with the loop, don't split
                    else:
                        # This is a sentence-ending punctuation
                        splits.append(current.strip())
                        current = ''
                else:
                    # This is a sentence-ending punctuation
                    splits.append(current.strip())
                    current = ''
        
        i += 1
    
    if current:  # Add any remaining text
        splits.append(current.strip())
    return splits

def is_valid_sentence(sentence):
    """Check if a sentence is valid (not empty, not just dashes or punctuation)"""
    if not sentence or not sentence.strip():
        return False
    
    # Remove whitespace for checking
    cleaned = sentence.strip()
    
    # Check if sentence is just dashes (any number of dashes)
    if all(c == '-' for c in cleaned):
        return False
    
    # Check if sentence is just punctuation and whitespace
    alphanumeric_chars = ''.join(c for c in cleaned if c.isalnum())
    return bool(alphanumeric_chars)

def process_section(section_text, section_type):
    """Process a section (thinking or answer) and return sentences with metadata"""
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

def process_response_to_sentences(response, apply_merging=True):
    """
    Complete pipeline to process a response into structured sentences.
    
    Args:
        response (str): The raw response text
        apply_merging (bool): Whether to apply colon and equals merging
    
    Returns:
        list: List of dictionaries with 'id', 'sentence', and 'type' keys
    """
    # Split response by </think> tag
    if '</think>' in response:
        parts = response.split('</think>', 1)
        thinking_part = parts[0].strip()
        answer_part = parts[1].strip()
        
        # Process thinking section
        thinking_sentences = process_section(thinking_part, 'think')
        
        # Process answer section  
        answer_sentences = process_section(answer_part, 'answer')
        
        # Combine all sentences
        all_sentences = thinking_sentences + answer_sentences
    else:
        # No </think> tag, treat entire response as answer section
        all_sentences = process_section(response, 'answer')
    
    # Apply post-processing if requested
    if apply_merging:
        processed_sentences = merge_colon_and_equals_sentences(all_sentences)
    else:
        processed_sentences = all_sentences
    
    # Create final result with sequential IDs
    result = []
    for i, sentence_data in enumerate(processed_sentences):
        result.append({
            'id': str(i),
            'sentence': sentence_data['sentence'],
            'type': sentence_data['type']
        })
    
    return result

# Post-process: merge sentences ending with ':' with next sentence and sentences starting with '=' with previous sentence
def merge_colon_and_equals_sentences(sentences):
    """Merge sentences ending with ':' with the next sentence and sentences starting with '=' with previous sentence"""
    # First pass: merge sentences ending with ':'
    merged = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i].copy()  # Make a copy to avoid modifying original
        current_sentence['sentence'] = current_sentence['sentence'].replace('<think>', '').strip()
        
        # Keep merging while current sentence ends with ':' and there's a next sentence
        while (current_sentence['sentence'].rstrip().endswith(':') and 
               i + 1 < len(sentences)):
            next_sentence = sentences[i + 1].copy()  # Make a copy
            next_sentence['sentence'] = next_sentence['sentence'].replace('<think>', '').strip()
            
            # Only merge if they have the same type (think/answer)
            if current_sentence['type'] == next_sentence['type']:
                # Merge the sentences
                current_sentence['sentence'] = current_sentence['sentence'] + ' ' + next_sentence['sentence']
                i += 1  # Move to next sentence (will be skipped in main loop)
            else:
                # Different types, can't merge
                break
        
        merged.append(current_sentence)
        i += 1
    
    # Second pass: merge sentences starting with '=' with previous sentence
    final_merged = []
    for i, sentence in enumerate(merged):
        sentence_copy = sentence.copy()
        sentence_copy['sentence'] = sentence_copy['sentence'].replace('<think>', '').strip()
        
        # Check if sentence starts with '='
        if (sentence_copy['sentence'].lstrip().startswith('=') and 
            len(final_merged) > 0 and 
            final_merged[-1]['type'] == sentence_copy['type']):
            # Merge with previous sentence
            final_merged[-1]['sentence'] = final_merged[-1]['sentence'] + ' ' + sentence_copy['sentence']
        else:
            # Add as new sentence
            final_merged.append(sentence_copy)
    
    return final_merged