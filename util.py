import numpy as np
from dataclasses import dataclass, fields, Field, is_dataclass, asdict
from typing import List, Any, Type, TypeVar, Tuple, Dict, Optional
import copy

# Define a generic type variable for the dataclass
T = TypeVar('T')

def _create_dataclass_instance_from_state(
    state_dict: Dict[str, Any],
    DataclassType: Type[T],
    original_field_types: Dict[str, Type],
    original_dtypes: Dict[str, np.dtype],
    keep_field_name: str
) -> Optional[T]:
    """Helper function to convert a state dictionary back to a dataclass instance."""
    final_data = {}
    all_field_names = [f.name for f in fields(DataclassType)]

    for field_name in all_field_names:
        if field_name not in state_dict:
            print(f"Warning: Field '{field_name}' missing in state_dict during conversion.")
            continue # Skip this field

        original_type = original_field_types.get(field_name)
        data = state_dict[field_name] # This is currently a list or other copied type

        if original_type is np.ndarray:
            try:
                dtype = original_dtypes.get(field_name)
                if keep_field_name == field_name or (dtype and 'bool' in str(dtype).lower()):
                    final_data[field_name] = np.array(data, dtype=bool)
                else:
                    final_data[field_name] = np.array(data, dtype=dtype)
            except Exception as e:
                 print(f"Warning: Could not convert field '{field_name}' back to numpy array: {e}. Keeping as list.")
                 final_data[field_name] = list(data) # Fallback to list
        elif original_type is list:
             final_data[field_name] = list(data) # Ensure it's a list
        else:
             final_data[field_name] = data # Keep other types as they were

    try:
        # Ensure all required fields are present before creating instance
        missing_fields = [fn for fn in all_field_names if fn not in final_data]
        if missing_fields:
            print(f"Error: Missing fields when creating dataclass instance: {missing_fields}")
            return None
        return DataclassType(**final_data)
    except Exception as e:
        print(f"Error creating final dataclass instance of type {DataclassType}: {e}")
        print("Data dictionary used:")
        print(final_data)
        return None


def aggregate_and_trace_results_generalized(iter_result: List[Optional[List[T]]], keep_field_name: str = "is_keep") -> Optional[List[Optional[T]]]:
    """
    Aggregates results across iterations and returns the state after each iteration.
    Handles None or empty lists in iter_result for iterations where no processing occurs.

    Args:
        iter_result: A list where each element is a list containing the dataclass
                     object(s) for that iteration step, OR None/[] if no processing occurred.
        keep_field_name: The name of the boolean field within the dataclass
                         that indicates whether to keep the item's state.

    Returns:
        A list of dataclass instances (or None if creation fails). Each instance
        represents the aggregated state after the corresponding iteration step.
        Returns None if initial input is invalid. The list length matches input length.
    """
    # Allow None in the input list type hint
    if not iter_result or not iter_result[0] or not iter_result[0][0]:
        print("Warning: Input iter_result is empty or first iteration invalid.")
        return None

    # --- Initialization ---
    try:
        # Check the first valid entry
        initial_log_list = iter_result[0]
        if not initial_log_list: # Should not happen based on check above, but safety
             print("Error: First iteration list is empty.")
             return None
        initial_log: T = initial_log_list[0]

        if not is_dataclass(initial_log):
             print(f"Error: Input object {type(initial_log)} is not a dataclass.")
             return None
        DataclassType: Type[T] = type(initial_log)
    except (IndexError, TypeError, AttributeError) as e:
        print(f"Error accessing initial data: {e}.")
        return None

    all_fields: Tuple[Field, ...] = fields(DataclassType)
    field_names: List[str] = [f.name for f in all_fields]

    if keep_field_name not in field_names:
        print(f"Error: keep_field_name '{keep_field_name}' not found.")
        return None

    try:
        initial_keep_status = getattr(initial_log, keep_field_name)
        if not isinstance(initial_keep_status, (list, np.ndarray)):
             print(f"Error: Field '{keep_field_name}' must be a list or numpy array.")
             return None
        num_slots = len(initial_keep_status)
    except (AttributeError, TypeError) as e:
        print(f"Error accessing or checking length of keep_field '{keep_field_name}': {e}")
        return None

    # Initialize the *current* state using lists (mutable copies)
    current_state = {}
    original_field_types = {}
    original_dtypes = {}
    initial_log_copy = copy.deepcopy(initial_log) # Deep copy for safety

    for field in all_fields:
        field_name = field.name
        try:
            initial_data = getattr(initial_log_copy, field_name) # Use the copy
            if isinstance(initial_data, np.ndarray):
                current_state[field_name] = initial_data.tolist()
                original_field_types[field_name] = np.ndarray
                original_dtypes[field_name] = initial_data.dtype
            elif isinstance(initial_data, list):
                current_state[field_name] = list(initial_data) # Mutable copy
                original_field_types[field_name] = list
            else:
                 current_state[field_name] = initial_data
                 original_field_types[field_name] = type(initial_data)
        except AttributeError:
             print(f"Error accessing initial data for field '{field_name}'.")
             return None

    # List to store the state snapshot after each iteration
    # Allow Optional[T] to handle potential conversion failures gracefully
    history_of_states: List[Optional[T]] = []

    # --- Store Initial State (State after Iteration 0) ---
    initial_state_instance = _create_dataclass_instance_from_state(
        copy.deepcopy(current_state), # Pass a copy
        DataclassType,
        original_field_types,
        original_dtypes,
        keep_field_name
    )
    # No need to fail entirely if initial conversion fails, just record None maybe?
    # For now, stick to failing early if initial state fails.
    if initial_state_instance is None:
         print("Error: Failed to create initial state instance.")
         return None
    history_of_states.append(initial_state_instance)

    # Track original indices of items needing updates
    indices_being_processed = [i for i, kept in enumerate(current_state[keep_field_name]) if not kept]
    processing_stopped = not indices_being_processed # Flag if processing already stopped

    # --- Iteration Loop ---
    for k in range(1, len(iter_result)):
        current_iter_input = iter_result[k]

        # --- Handle None or Empty Input for Iteration k ---
        if processing_stopped or current_iter_input is None or not current_iter_input:
            #print(f"Iteration {k}: No processing data or processing already stopped. Copying previous state.")
            if history_of_states: # Ensure there's a previous state to copy
                 last_state = history_of_states[-1]
                 # If last state was None due to previous error, propagate None
                 history_of_states.append(copy.deepcopy(last_state) if last_state else None)
            else: # Should not happen if initial state worked
                 history_of_states.append(None)
            processing_stopped = True # Ensure subsequent iterations also copy state
            continue # Skip to next iteration

        # --- Process Valid Data for Iteration k ---
        try:
            current_log: T = current_iter_input[0] # Get the dataclass instance
            if type(current_log) is not DataclassType:
                print(f"Warning: Dataclass type mismatch in iteration {k}. Copying previous state.")
                if history_of_states: history_of_states.append(copy.deepcopy(history_of_states[-1]))
                else: history_of_states.append(None)
                continue
        except IndexError:
            print(f"Warning: Empty list found in iter_result[{k}] when data was expected. Copying previous state.")
            if history_of_states: history_of_states.append(copy.deepcopy(history_of_states[-1]))
            else: history_of_states.append(None)
            continue

        try:
            current_keep_status = getattr(current_log, keep_field_name)
            num_processed_in_iter = len(current_keep_status)
        except (AttributeError, TypeError) as e:
             print(f"Error accessing keep status in iteration {k}: {e}. Copying previous state.")
             if history_of_states: history_of_states.append(copy.deepcopy(history_of_states[-1]))
             else: history_of_states.append(None)
             continue

        # Sanity Check
        if len(indices_being_processed) != num_processed_in_iter:
             print(f"Warning: Mismatch in expected ({len(indices_being_processed)}) vs actual "
                   f"({num_processed_in_iter}) items processed in iteration {k}. Copying previous state and stopping updates.")
             if history_of_states: history_of_states.append(copy.deepcopy(history_of_states[-1]))
             else: history_of_states.append(None)
             processing_stopped = True # Stop further processing attempts
             continue # Proceed to next iteration index, but just copy state

        # --- Update State Based on current_log ---
        new_indices_being_processed = []
        for j in range(num_processed_in_iter):
            original_idx = indices_being_processed[j]
            for field_name in field_names:
                if field_name not in current_state: continue
                try:
                    current_data_field = getattr(current_log, field_name)
                    value_to_update = current_data_field[j]
                    current_state[field_name][original_idx] = value_to_update
                except (AttributeError, IndexError, TypeError) as e:
                     print(f"Warning: Could not update field '{field_name}' for original index {original_idx} "
                           f"from iteration {k}, item {j}: {e}")

            if not current_state[keep_field_name][original_idx]:
                new_indices_being_processed.append(original_idx)

        indices_being_processed = new_indices_being_processed
        if not indices_being_processed:
            processing_stopped = True # Mark processing as stopped for future iterations

        # --- Store State Snapshot After Iteration k ---
        state_after_iter_k = _create_dataclass_instance_from_state(
            copy.deepcopy(current_state), # Pass a copy
            DataclassType,
            original_field_types,
            original_dtypes,
            keep_field_name
        )
        history_of_states.append(state_after_iter_k) # Append instance or None if creation failed


    # Final check: Ensure history length matches input length
    # This might be redundant now with the loop handling None/empty/stopped cases, but keep as safeguard
    while len(history_of_states) < len(iter_result):
         print(f"Padding history at the end for iteration {len(history_of_states)}")
         if history_of_states: # Check if history has elements
            last_state = history_of_states[-1]
            history_of_states.append(copy.deepcopy(last_state) if last_state else None)
         else: # Should only happen if input was completely invalid
             history_of_states.append(None)


    return history_of_states
