from typing import Any, Union, Dict
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def check_attribute_recall(model: Any, prompt: str, target_attribute: str) -> bool:
    """
    Runs a clean trace of the model to verify if it correctly predicts the target attribute.
    
    Args:
        model: The LanguageModel.
        prompt (str): The clean factual recall prompt.
        target_attribute (str): The expected output token(s).
    Returns:
        bool: True if the model's prediction matches the target attribute.
    """
    with model.trace(prompt):
        last_predicted_id = model.lm_head.output.argmax(dim=-1)[0][-1].save()
    
    return compare_pred_and_target(model, last_predicted_id, target_attribute)


def compare_pred_and_target(model: Any, last_predicted_id: int, target_attribute: str) -> bool:
    """
    Runs a clean trace of the model to verify if it correctly predicts the target attribute.
    
    Args:
        model: The LanguageModel.
        last_predicted_id (int): The ID of the last predicted token.
        target_attribute (str): The expected output token(s).
        
    Returns:
        bool: True if the model's prediction matches the target attribute.
    """
        
    prediction = model.tokenizer.decode(last_predicted_id)
    
    is_correct = (prediction.strip() == target_attribute.strip())

    if is_correct:
        print(f"Success: Predicted '{prediction}' matches '{target_attribute}'")
    else:
        print(f"Failed: Predicted '{prediction}' instead of '{target_attribute}'")
        
    return is_correct



def collect_layer_inputs(model: Any, prompt: str, ent_slice: Union[slice, list, int]) -> Dict[int, Any]:
    activations = {}

    with model.trace(prompt):
        for layer_idx in range(len(model.model.layers)):
            inp = model.model.layers[layer_idx].input[:, ent_slice, :].save()
            activations[layer_idx] = inp
    
    return activations



def get_entity_slice(prompt: str, entity: str, tokenizer):

    # Step 1: Get offset mappings from the tokenizer
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    input_ids = encoding.input_ids[0]
    offsets = encoding.offset_mapping[0]  # list of (start_char, end_char) for each token

    # print all tokens as decided strings with their index in the prompt
    # for i, (start, end) in enumerate(offsets.tolist()):
    #     token_str = tokenizer.decode([input_ids[i]])
    #     print(f"Token index: {i}, Token: '{token_str}', Char span: ({start}, {end})")

    # Step 2: Find the character span of the entity in the prompt
    start_char = prompt.lower().find(entity.lower())
    if start_char == -1:
        print(f"Entity '{entity}' not found in the prompt.")
        return None
    end_char = start_char + len(entity)

    if start_char != 0:
        start_char -= 1 # Adjust for the space before the entity
    #print(f"Entity '{entity}' found at character span: ({start_char}, {end_char})")


    # Step 3: Match character span to token span using offsets
    entity_token_idxs = [
        i for i, (start, end) in enumerate(offsets.tolist())
        if start >= start_char and end <= end_char
    ]

    # Remove <|begin_of_text|> from the indices in llama case.
    if "llama" in tokenizer.name_or_path and start_char == 0:
        #print("Removing <|begin_of_text|> token from entity indices")
        entity_token_idxs = entity_token_idxs[1:]

    if not entity_token_idxs:
        print(f"Failed to find token indexes for entity '{entity}'.")
        return None

    # print the last entity token that we found by its index
    tokens = tokenizer.decode(input_ids[entity_token_idxs])
    # print(f"Tokens of the entity '{entity}': '{tokens}' - {len(entity_token_idxs)} tokens found.")
    # print(entity_token_idxs)

    if entity != tokens and " " + entity != tokens:
        print(f"Warning: Extracted tokens '{tokens}' do not match the entity '{entity}'.")
        return None
    
    #print(entity_token_idxs)
    #return entity_token_idxs
    return slice(entity_token_idxs[0], entity_token_idxs[-1] + 1)


def get_counterfact_entity(model, prompt_template, entity, relation_id, target_token, entity_slice, counterfact_raw_data):

    if entity_slice is None:
        print(f"Cannot get counterfact entity because of tokenization issue with entity '{entity}'.")
        return None 

    for item in counterfact_raw_data:
        info = item["requested_rewrite"]
        if info["relation_id"] == relation_id and info["target_true"]["str"] != target_token:
            counterfact_entity = info["subject"]
            counterfact_prompt = prompt_template.format(counterfact_entity)
            counterfact_ent_slice = get_entity_slice(counterfact_prompt, counterfact_entity, model.tokenizer)

            # Check if the corrupted entity has the same number of tokens as the original entity
            if counterfact_ent_slice and counterfact_ent_slice == entity_slice and not check_attribute_recall(model, counterfact_prompt, target_token):
                return counterfact_entity
    return None