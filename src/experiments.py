from typing import List, Any
from .patching import isolate_path, lock_activation

def representation_knockout(
    edited_model: Any,
    ent_slice: Union[slice, list, int],
    path_inputs: Dict[int, Any], 
    counterfactual_inputs: Dict[int, Any],
    l_attr: int = None
) -> None:
    """
    Overwrites the representation at l_attr output with a counterfactual representation.
    
    Args:
        model: The LanguageModel.
        l_attr (int): The layer index of the attribute representation.
        ent_slice (slice | list | int): The sequence position(s) of the entity.
        counterfactual_inputs (Dict[int, Any]): Counterfactual representations for each layer.
    """
    isolate_path(edited_model, path_inputs, counterfactual_inputs, ent_slice, l_attr)

    # Overwrite the representation at l_attr output with the counterfactual
    if l_attr < len(edited_model.model.layers) - 1:
        edited_model.model.layers[l_attr + 1].input[:, ent_slice, :] = counterfactual_inputs[l_attr + 1]


def downstream_injection(
    edited_model: Any,
    ent_slice: Union[slice, list, int],
    counterfactual_inputs: Dict[int, Any],
    l_attr: int,
    l_attr_rep: Any,
) -> None:
    """
    Injects the representation at l_attr output into all subsequent layers, 
    while making all previous layers counterfactual.
    """
    # Overwrite all previous layers with counterfactuals
    edited_model.model.layers[0].input[:, ent_slice, :] = counterfactual_inputs[0]

    # Inject the representation at l_attr output into all subsequent layers
    for layer_idx in range(l_attr + 1, len(edited_model.model.layers)):
        edited_model.model.layers[layer_idx].input[:, ent_slice, :] = l_attr_rep 


def path_and_continuation(
    edited_model: Any,
    ent_slice: Union[slice, list, int],
    counterfactual_inputs: Dict[int, Any],
    path_layers_indices: List[int],
    l_attr_rep: Any,
) -> None:
    """
    Injects the representation at l_attr output into all path and higher layers.
    """
    path_inputs_overwrite = {layer_idx: l_attr_rep for layer_idx in path_layers_indices}

    # Overwrite all path layer inputs with l_attr representation
    isolate_path(edited_model, path_inputs_overwrite, counterfactual_inputs, ent_slice, l_attr)

    # lock l_attr output to all subsequent layers
    lock_activation(edited_model, l_attr_rep, l_attr, len(edited_model.model.layers) - 1, ent_slice)


def global_broadcast(
    edited_model: Any,
    ent_slice: Union[slice, list, int],
    l_attr_rep: Any,
) -> None:
    """
    Injects the representation at l_attr output into all model layers.
    """
    for layer_idx in range(len(edited_model.model.layers)):
        edited_model.model.layers[layer_idx].input[:, ent_slice, :] = l_attr_rep


def run_sufficiency_test(
    model: Any, 
    prompt: str,
    expected_token: str,
    ent_slice: Union[slice, list, int], 
    path_inputs: Dict[int, Any],
    counterfactual_inputs: Dict[int, Any], 
    l_attr_rep: Any,
    intervention_type: str
) -> bool:
    """
    Runs one of the four targeted patching interventions to test the
    necessity and sufficiency of l_attr output.
    
    Valid intervention_types
    - "representation_knockout": Overwrite representation after l_attr with counterfactual.
    - "downstream_injection": Inject l_attr representation into all subsequent layers,
       while making all previous layers counterfactual
    - "path_continuation": Inject l_attr into all path and higher layers.
    - "global_broadcast": Inject l_attr into all model layers.
    
    Returns:
        bool: True if the model correctly predicted the target attribute, False otherwise.
    """
    l_attr = max(path_inputs.keys())  # The last layer in the path is l_attr
    with model.trace(prompt):
        if intervention_type == "representation_knockout":
            representation_knockout(model, ent_slice, path_inputs, counterfactual_inputs, l_attr)
        elif intervention_type == "downstream_injection":
            downstream_injection(model, ent_slice, counterfactual_inputs, l_attr, l_attr_rep)
        elif intervention_type == "path_continuation":
            path_and_continuation(model, ent_slice, counterfactual_inputs, list(path_inputs.keys()), l_attr_rep)
        elif intervention_type == "global_broadcast":
            global_broadcast(model, ent_slice, l_attr_rep)
        else:
            raise ValueError(f"Invalid intervention_type: {intervention_type}")

        # Get the final token prediction
        last_predicted_id = model.lm_head.output[0, -1, :].argmax(dim=-1).save()

    return compare_pred_and_target(model, last_predicted_id, expected_token)
            