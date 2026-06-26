from typing import Dict, Any, Union, Tuple, List
from .patching import isolate_path, lock_activation
from .utils import *

def find_l_attr(
    model: Any, 
    prompt: str, 
    expected_token: str, 
    ent_slice: Union[slice, list, int]
) -> int:
    """
    Finds l_attr: the earliest layer whose output is sufficient to predict 
    the target attribute when broadcast to all subsequent layers.
    
    Args:
        model: The LanguageModel.
        prompt (str): The clean factual recall prompt.
        expected_token (str): The target attribute token.
        ent_slice (slice | list | int): The sequence position(s) of the entity.
        
    Returns:
        int: The layer index of l_attr (-1 represents the embedding layer).
    """
    num_layers = len(model.model.layers)
    
    # Iterate from embeddings (-1) up to the second-to-last layer.
    for candidate_l in range(-1, num_layers - 1):
        
        # The output of candidate_l is the input to candidate_l + 1
        next_layer = candidate_l + 1
        
        with model.trace(prompt):
            # Grab the clean representation flowing into the next layer
            rep_to_inject = model.model.layers[next_layer].input[:, ent_slice, :]
            
            # Lock this clean representation
            lock_activation(model, rep_to_inject, next_layer, num_layers-1, ent_slice)
            
            # Predict the final token
            last_predicted_id = model.lm_head.output[0, -1, :].argmax(dim=-1).save()
            
        if compare_pred_and_target(model, last_predicted_id, expected_token):
            print(f"Found l_attr at layer {candidate_l}.")
            return candidate_l

    # If the loop finishes without returning, no early layer was sufficient.
    print(f"No early sufficiency. l_attr defaults to final layer: {num_layers - 1}")
    return num_layers - 1


def test_path_extension(
    model: Any,
    prompt: str,
    expected_token: str,
    ent_slice: Union[slice, list, int],
    path_inputs: Dict[int, Any],
    counterfactual_inputs: Dict[int, Any],
    candidate_layer: int,
    upper_bound: int,
    run_isolation_only_check: bool = True
) -> Tuple[bool, Any]:
    """
    Tests if a prefix is valid by composing isolate() and lock().
    """
    # Step 1: ISOLATE the path and check if the attribute is retrieved
    if run_isolation_only_check:
        with model.trace(prompt):
            
            # ISOLATE
            isolate_path(model, path_inputs, counterfactual_inputs, ent_slice, candidate_layer)
            last_predicted_id = model.lm_head.output.argmax(dim=-1)[0][-1].save()
        
        # Check if this composition successfully retrieved the attribute
        is_correct = compare_pred_and_target(model, last_predicted_id, expected_token)

        if not is_correct:
            return False, None
    
    # step 2: ISOLATE the path AND LOCK the upper bound and check if the attribute is still retrieved
    with model.trace(prompt):
        
        isolate_path(model, path_inputs, counterfactual_inputs, ent_slice, candidate_layer)

        # Extract the candidate layer's output
        candidate_output = model.model.layers[candidate_layer].output[0][ent_slice, :].save()
        
        # LOCK: Extract upper bound output and override all subsequent layers
        next_after_upper = upper_bound + 1
        last_layer = len(model.model.layers) - 1

        if next_after_upper <= last_layer:
            upper_bound_output = model.model.layers[next_after_upper].input[:, ent_slice, :]

            # Lock all layers from next_after_upper to the end of the model
            lock_activation(model, upper_bound_output, next_after_upper, last_layer, ent_slice)
        
        last_predicted_id = model.lm_head.output.argmax(dim=-1)[0][-1].save()
        
    # Check if this composition successfully retrieved the attribute
    is_correct = compare_pred_and_target(model, last_predicted_id, expected_token)

    return is_correct, candidate_output



def find_minimal_path(
    model: Any, 
    clean_prompt: str, 
    counterfact_prompt: str, 
    expected_token: str, 
    ent_slice: Union[slice, list, int],
    upper_bound: int
) -> List[int]:
    """
    Constructs a minimal computational path via an iterative greedy search.
    Maximizes the number of skipped layers at every iteration.
    
    Args:
        upper_bound (int): l_attr for primary paths, or final layer for alternative paths.
        
    Returns:
        List[int]: The sequence of necessary layer indices forming the minimal path.
    """
    # 1. Pre-compute counterfactual inputs for the intermediate skipped layers
    counterfactual_inputs = collect_layer_inputs(model, counterfact_prompt, ent_slice)
    clean_inputs = collect_layer_inputs(model, clean_prompt, ent_slice)
    
    path_layers = [-1] # Start with the embedding layer (conceptually -1)
    path_inputs = {}
    current_layer = -1
    path_prefix_output = clean_inputs[0] # Start with the clean embedding as the initial path input
    
    # Check if embeddings alone are sufficient
    is_sufficient, _ = test_path_extension(
        model, clean_prompt, expected_token, ent_slice, 
        path_inputs, counterfactual_inputs, 0, -1
    )
    if is_sufficient:
        return path_layers # Path is empty, embeddings are sufficient

    # 2. Iterative Greedy Search
    while current_layer < upper_bound:
        jump_successful = False
        
        # Try the largest possible jump first, walking backwards to the current layer
        for candidate_layer in range(upper_bound, current_layer, -1):
            path_inputs[candidate_layer] = path_prefix_output # The last valid path output is the input for the candidate layer

            is_valid_prefix, candidate_output = test_path_extension(
                model, counterfact_prompt, expected_token, ent_slice, 
                path_inputs, counterfactual_inputs, candidate_layer, upper_bound
            )
            
            if is_valid_prefix:
                path_layers.append(candidate_layer)
                path_prefix_output = candidate_output
                current_layer = candidate_layer
                jump_successful = True

                if candidate_layer == upper_bound:
                    return path_layers
                
                # Check if this new candidate layer is sufficient on its own to be the new upper bound
                is_sufficient, _ = test_path_extension(
                    model, counterfact_prompt, expected_token, ent_slice, 
                    path_inputs, counterfactual_inputs, 
                    candidate_layer=candidate_layer, 
                    upper_bound=candidate_layer,
                    run_isolation_only_check=False
                )

                if is_sufficient:
                    print(f"Early sufficiency reached at layer {candidate_layer}. Stopping search.")
                    return path_layers
                
                break

            del path_inputs[candidate_layer] # Clean up for the next candidate

        if not jump_successful:
            # If no jump works, the path is broken. 
            # In a robust model, this shouldn't happen if upper_bound was truly sufficient.
            print(f"Warning: Path broken at layer {current_layer}. No valid subsequent jumps found.")
            break
            
    return path_layers