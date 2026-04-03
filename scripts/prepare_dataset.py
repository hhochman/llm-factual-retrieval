import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import requests
from src.loaders import load_model
from src.utils import check_attribute_recall, get_entity_slice, get_counterfact_entity

def download_dataset_if_missing(url: str, save_path: str):
    """Downloads the dataset and saves it locally if it doesn't exist."""
    if os.path.exists(save_path):
        print(f"Found local dataset at {save_path}. Skipping download.")
        return

    print(f"Downloading raw dataset from {url}...")
    response = requests.get(url)
    
    # This raises an error if the URL is broken (e.g., a 404 or 500 error)
    response.raise_for_status() 
    
    # Ensure the target directory (like 'data/') actually exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")


def main():

    # Example model names: 'meta-llama/Llama-3.1-8B', 'Qwen/Qwen3-8B'
    # Example output path: 'data/Llama-3.1-8B/counterfact_sample.json'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, default="data/counterfact.json")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--target_count", type=int, default=2000)
    args = parser.parse_args()

    url = "https://rome.baulab.info/data/dsets/counterfact.json"
    download_dataset_if_missing(url, args.raw_data_path)

    model = load_model(args.model_name)
    tokenizer = model.tokenizer
    
    with open(args.raw_data_path, 'r') as f:
        raw_data = json.load(f)

    valid_dataset = []
    
    for item in raw_data:
        if len(valid_dataset) >= args.target_count:
            break

        case_id = item['case_id']
        item_info = item['requested_rewrite']    
        prompt_template = item_info['prompt']
        entity = item_info['subject']
        target_token = item_info['target_true']['str']
        relation_id = item_info['relation_id']

        prompt = prompt_template.format(entity)
        
        # Check if the model know the fact
        if check_attribute_recall(model, prompt, target_token):

            # Get the slice
            ent_slice = get_entity_slice(prompt, entity, tokenizer)

            # Get a counterfact entity
            counterfact_entity = get_counterfact_entity(model, prompt_template, entity, relation_id, target_token, ent_slice, raw_data)

            if counterfact_entity is None:
                print(f"Skipping case {case_id} - counterfactual entity not found.")
                continue
            
            # Save to our clean dataset
            valid_dataset.append({
                "case_id": case_id,
                "entity": entity,
                "counterfact_entity": counterfact_entity,
                "relation_id": relation_id,
                "prompt": prompt_template,
                "target_token": target_token,
                "entity_slice_start": ent_slice.start,
                "entity_slice_stop": ent_slice.stop
            })
            print(f"Added {len(valid_dataset)}/{args.target_count}: {entity}")
                


    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(valid_dataset, f, indent=4)
    print(f"Saved {len(valid_dataset)} valid prompts to {args.output_path}")

if __name__ == "__main__":
    main()