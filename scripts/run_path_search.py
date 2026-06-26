import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import logging
from src.loaders import load_model
from src.search import find_l_attr, find_minimal_path
from src.utils import set_seed
from tqdm import tqdm



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Extract minimal computation paths for Factual Retrieval")
    parser.add_argument("--model", type=str, choices=["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B"], required=True)
    parser.add_argument("--mode", type=str, choices=["primary", "alternative"], required=True, 
                        help="primary uses l_attr as bound; alternative uses final layer.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    logging.info(f"Loading model {args.model}...")
    model = load_model(args.model)
    
    logging.info(f"Loading data from {args.data_path}...")
    dataset = json.load(open(args.data_path, 'r'))

    

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'a') as f:
        for item in tqdm(dataset, desc="Processing items"):
            case_id = item['case_id']
            entity = item['entity']
            counterfact_entity = item['counterfact_entity']
            prompt_template = item['prompt']
            target_token = item['target_token']
            ent_slice = slice(item['entity_slice_start'], item['entity_slice_stop'])

            prompt = prompt_template.format(entity)
            counterfact_prompt = prompt_template.format(counterfact_entity)

            l_attr = None

            if args.mode == "primary":
                l_attr = find_l_attr(model, prompt, target_token, ent_slice)
                upper_bound = l_attr
            else:
                upper_bound = len(model.model.layers) - 1

            path = find_minimal_path(model, prompt, counterfact_prompt, target_token, ent_slice, upper_bound)

            if l_attr is not None:
                result = {"case_id": item['case_id'], "l_attr": l_attr, "path": path}
            else:
                result = {"case_id": item['case_id'], "path": path}

            f.write(json.dumps(result) + '\n')

    logging.info(f"Finished processing. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()