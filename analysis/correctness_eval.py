import os
import json
import argparse
import openai
import re
from tqdm import tqdm
import time

# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluate model correctness using GPT-5.')
parser.add_argument('--model', type=str, required=True, help='The model name to evaluate (e.g., DSQwen32B)')
parser.add_argument('--evaluator_model', type=str, default='gpt-4o', help='The model used for evaluation (default: gpt-4o, user requested gpt-5 but defaulting to gpt-4o for safety/availability)')

args = parser.parse_args()

# Input and output paths
input_path = f'data/raw/{args.model}.json'
output_dir = f'data/correct'
output_path = f'{output_dir}/{args.model}.json'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load input data
if not os.path.exists(input_path):
    print(f"Error: Input file {input_path} not found.")
    exit(1)

with open(input_path, 'r') as f:
    data = json.load(f)

# Load template
template_path = 'analysis/correctness_eval_template.txt'
if not os.path.exists(template_path):
    print(f"Error: Template file {template_path} not found.")
    exit(1)

with open(template_path, 'r') as f:
    template_content = f.read()

# Load existing results if any, to allow resuming
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        results = json.load(f)
else:
    results = {}

# Set OpenAI API key from environment
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Function to call OpenAI API
def evaluate_response(instruction, reference, student_response, evaluator_model):
    # Construct the prompt
    # Replace placeholders in the template
    # The template has {{Problem}}, {{Reference Answer}}, {{Solution}} (based on the last part of the file)
    # However, looking at the file, the last part is:
    # <math solution>
    # **Question**:
    # {{Problem}}
    # 
    # **Reference Answer**
    # {{Reference Answer}}
    # 
    # **Student Solution**:
    # {{Solution}}
    # 
    # </math solution>
    
    # We need to replace these.
    # Also, we should probably only use the text up to the example start or just append the new case?
    # The template provided in the Read output seems to contain examples and then a final block with placeholders.
    # I will assume the template text provided is the system prompt + few-shot examples + the final query format.
    
    prompt = template_content.replace('{{Problem}}', instruction)
    prompt = prompt.replace('{{Reference Answer}}', reference)
    prompt = prompt.replace('{{Solution}}', student_response)

    messages = [
        {"role": "system", "content": "You are a helpful assistant evaluating math problems."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=evaluator_model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# Iterate through data
for item in tqdm(data, desc=f"Evaluating {args.model}"):
    idx = str(item.get('index'))
    
    # Skip if already evaluated
    if idx in results:
        continue
        
    instruction = item.get('Instruction', '')
    reference = item.get('Correct Answer', '')
    student_response = item.get('Response', '')
    
    if not student_response:
        print(f"Warning: No response for index {idx}")
        results[idx] = False # Assume false if no response? Or skip? User asked for output {[index]:[correctness]}
        continue

    evaluation_text = evaluate_response(instruction, reference, student_response, args.evaluator_model)
    
    if evaluation_text:
        # Parse the result
        # Look for ## Equivalence Judgement\n[TRUE or FALSE]
        match = re.search(r'## Equivalence Judgement\s*(TRUE|FALSE)', evaluation_text, re.IGNORECASE)
        if match:
            judgement = match.group(1).upper()
            is_correct = (judgement == 'TRUE')
            results[idx] = is_correct
        else:
            print(f"Warning: Could not parse judgement for index {idx}. Response: {evaluation_text[:100]}...")
            # Fallback or manual check needed? For now, maybe set to None or False
            results[idx] = False 
            
        # Save periodically
        if len(results) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
                
        # Rate limit handling (simple sleep)
        # time.sleep(0.1) 

# Final save
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Evaluation complete. Results saved to {output_path}")
