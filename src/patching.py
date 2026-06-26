from typing import Dict, Any, Union


def isolate_path(
    edited_model: Any, 
    path_inputs: Dict[int, Any], 
    counterfactual_inputs: Dict[int, Any], 
    ent_slice: Union[slice, list, int],
    candidate_layer: int
) -> None:
    """Applies the isolate operation to the active edited_model."""
    for layer_idx, inp in path_inputs.items():
        # Path layers receive specific inputs
        edited_model.model.layers[layer_idx].input[:, ent_slice, :] = inp
        
        # Intermediate layers receive counterfactuals to sever the natural flow
        if layer_idx < candidate_layer:
            next_layer_idx = layer_idx + 1
            edited_model.model.layers[next_layer_idx].input[:, ent_slice, :] = counterfactual_inputs[next_layer_idx]


def lock_activation(
    edited_model: Any, 
    activation_to_inject: Any, 
    start_layer_idx: int, 
    end_layer_idx: int, 
    ent_slice: Union[slice, list, int]
) -> None:
    """Applies the lock operation to the active edited_model."""
    for layer_idx in range(start_layer_idx, end_layer_idx + 1):
        # print(f"Locking layer {layer_idx} with injected activation.")
        edited_model.model.layers[layer_idx].output[0][ent_slice, : ] = activation_to_inject