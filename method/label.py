from openai import OpenAI
import os
import json
from tqdm import tqdm
from method.utils import process_new_data
import argparse
from multiprocessing import Pool
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--annotate_model', type=str, default='gpt-5')
parser.add_argument('--response_model', type=str, default='deepseekR1')
args = parser.parse_args()

def process_item(item_data, idx, model, guidebook=True):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    max_retry = 10
    retry_count = 0
    while retry_count < max_retry:
        try:
            process_new_data(client, item_data, sample_index=idx, model=model, 
                        guidebook=guidebook, output_path=f'data/label/{args.response_model}/{args.annotate_model}')
            break
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            retry_count += 1
            continue

def check_exist(idx):
    file_path = f"data/label/{args.response_model}/{args.annotate_model}/{idx + 1}.json"
    if os.path.exists(file_path):
        return True
    else:
        return False

def main():
    new_data = []
    with open(f"data/raw/{args.response_model}.json", "r") as f:
        new_data = json.load(f)

    for idx, data in enumerate(new_data):
        if not check_exist(idx):
            print(f"Processing {idx}...")
            process_item(data, idx, args.annotate_model, True)


if __name__ == '__main__':
    main()
