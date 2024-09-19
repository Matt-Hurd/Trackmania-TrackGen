from typing import Any, Dict, List

import jax
import numpy as np
import jax.numpy as jnp
import json

from data_manager import TrackmaniaDataManager
from enums import EncodingType, EventType
from features import Features
from main import BasicTrackmaniaNN, DataProcessor, ModelConfig, create_learning_rate_fn, create_train_state, restore_train_state

def autoregressive_predict(
    model,
    initial_input: jnp.ndarray,
    params: Any,
    block_data: Dict[str, jnp.ndarray],
    output_length: int = 100,
    train: bool = False,
    rngs: Dict[str, jax.random.PRNGKey] = None,
    window_size: int = 32
) -> Dict[str, jnp.ndarray]:
    """
    Perform autoregressive prediction using a fixed-size sliding window for both input sequences and block data.
    
    Args:
        model: The trained JAX model.
        initial_input (jnp.ndarray): Initial input sequence with shape (batch_size, sequence_length, num_features).
        params: Model parameters.
        block_data (Dict[str, jnp.ndarray], optional): Dictionary of block-related data.
        output_length (int): Total number of events to predict.
        train (bool, optional): Whether the model is in training mode.
        rngs (Dict[str, jax.random.PRNGKey], optional): Random number generators.
        window_size (int): The fixed number of events to maintain in the input window.
    
    Returns:
        Dict[str, jnp.ndarray]: A dictionary containing the complete predicted sequence and generated block data.
    """
    
    generated_blocks = {
        feature.name: jnp.copy(block_data[feature.name]) for feature in Features.get_block_features() if feature.encoding != EncodingType.NONE
        # feature.name: jnp.zeros(
        #     (block_data[feature.name].shape[0], 0, block_data[feature.name].shape[2]) if len(block_data[feature.name].shape) == 3 else (block_data[feature.name].shape[0], 0),
        #     dtype=jnp.int32 if feature.encoding == EncodingType.TOKENIZED else jnp.float32
        # )
        # for feature in Features.get_block_features() if feature.encoding != EncodingType.NONE
    }

    predicted_sequence = initial_input  # Shape: (batch_size, window_size, num_features)
    generated_events = []
    current_input = initial_input

    # DEBUG, ADD ALL INPUTS TO GENERATED EVENTS
    for i in range(initial_input.shape[1]):
        generated_events.append(initial_input[:, i:i+1, :])

    for step in range(output_length):
        # Apply the model to the current window
        next_step = model.apply(
            {'params': params},
            current_input,
            block_data=block_data,
            train=train,
            rngs=rngs
        )
        
        next_step_prediction = []

        # Collect predictions for each feature
        for feature in Features.get_all_features():
            if feature.encoding == EncodingType.NONE:
                continue

            # Extract the prediction for the last time step
            pred = next_step[feature.name][:, -1:, :]  # Shape: (batch_size, 1, feature_dim)

            if feature.is_block_feature:
                # Handle block feature predictions
                if feature.encoding == EncodingType.TOKENIZED:
                    pred_token = jnp.argmax(pred, axis=-1)  # Shape: (batch_size, 1)
                    new_block = pred_token.astype(jnp.int32)  # Correct dtype
                elif feature.encoding == EncodingType.ONE_HOT:
                    pred_label = jnp.argmax(pred, axis=-1)  # Shape: (batch_size, 1)
                    new_block = jax.nn.one_hot(pred_label, num_classes=feature.size).astype(jnp.float32)  # Shape: (batch_size, 1, feature.size)
                else:
                    new_block = pred  # Assume pred is already in the correct format

                # Concatenate the new block data
                updated_block = jnp.concatenate([block_data[feature.name], new_block], axis=1)  # Shape: (batch_size, current+1, ...)

                if updated_block.shape[1] > window_size:
                    # Extract the overflow to generated_blocks
                    if len(updated_block.shape) == 3:
                        overflow = updated_block[:, :-window_size, :]  # Shape: (batch_size, overflow_length, ...)
                    else:
                        overflow = updated_block[:, :-window_size]
                    # Concatenate the overflow to the corresponding generated_blocks entry
                    generated_blocks[feature.name] = jnp.concatenate([generated_blocks[feature.name], overflow], axis=1)
                    # Retain only the latest 'window_size' entries in block_data
                    if len(updated_block.shape) == 3:
                        block_data[feature.name] = updated_block[:, -window_size:, :]
                    else:
                        block_data[feature.name] = updated_block[:, -window_size:]
            else:
                # Collect predictions for non-block features
                next_step_prediction.append(pred)

        if next_step_prediction:
            # Concatenate predictions from all non-block features along the feature dimension
            next_step_concat = jnp.concatenate(next_step_prediction, axis=-1)  # Shape: (batch_size, 1, total_feature_dim)
            # Append the new predictions to the predicted_sequence
            predicted_sequence = jnp.concatenate([predicted_sequence, next_step_concat], axis=1)  # Shape: (batch_size, window_size +1, total_feature_dim)
            # Store the generated events
            generated_events.append(next_step_concat)

            # Maintain the sliding window for the input sequence
            predicted_sequence = predicted_sequence[:, -window_size:, :]  # Shape: (batch_size, window_size, total_feature_dim)

            # Update current_input for the next prediction step
            current_input = predicted_sequence
        else:
            # If there are no non-block features, simply continue maintaining the window
            if predicted_sequence.shape[1] > window_size:
                predicted_sequence = predicted_sequence[:, -window_size:, :]
            current_input = predicted_sequence

    # Concatenate all generated events along the sequence dimension
    if generated_events:
        generated_events = jnp.concatenate(generated_events, axis=1)  # Shape: (batch_size, output_length, total_feature_dim)
    else:
        # If no events were generated, return an empty array
        generated_events = jnp.empty((initial_input.shape[0], 0, initial_input.shape[2]), dtype=jnp.float32)

    return {
        'events': generated_events,
        'blocks': generated_blocks
    }

def get_inverse_tokenizers(tokenizers):
    inverse_tokenizers = {}
    for field, tokenizer in tokenizers.items():
        inverse_tokenizers[field] = {idx: value for value, idx in tokenizer.items()}
    return inverse_tokenizers

def save_predictions_as_json(
    predictions: np.ndarray,  # Changed to numpy.ndarray since we convert earlier
    block_data: Dict[str, np.ndarray],
    filename: str,
    data_processor: DataProcessor
):
    """
    Save model predictions and block data as a JSON file.

    Args:
        predictions (np.ndarray): The predicted sequence with shape (batch_size, sequence_length, num_features).
        block_data (Dict[str, np.ndarray]): Block-related data for features.
        filename (str): The destination filename for the JSON output.
    """
    events = []

    predictions = np.array(predictions)
    block_data = {key: np.array(value) for key, value in block_data.items()}

    untokenizers = get_inverse_tokenizers(data_processor.tokenizers)

    batch_size, sequence_length, num_features = predictions.shape

    print(batch_size, sequence_length, num_features)

    if batch_size != 1:
        raise ValueError("This function currently supports only batch_size=1.")

    def convert_value(value):
        if isinstance(value, (np.integer, np.int_)):
            return int(value)
        elif isinstance(value, (np.floating, np.float_)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value

    for x in range(sequence_length):
        event = {}

        event_type_feature = Features.EVENT_TYPE
        event_type_index = Features.get_feature_index(event_type_feature, False)
        raw_event_type = predictions[0, x, event_type_index]

        if event_type_feature.encoding == EncodingType.ONE_HOT:
            event_type = int(np.argmax(raw_event_type))
        elif event_type_feature.encoding == EncodingType.TOKENIZED:
            event_type = int(raw_event_type)
        else:
            event_type = convert_value(raw_event_type)

        try:
            event_type_enum = EventType(event_type)
            event['event_type'] = event_type_enum.name
        except ValueError:
            event['event_type'] = f"UNKNOWN_EVENT_TYPE_{event_type}"

        for feature in Features.get_all_features():
            if feature.encoding == EncodingType.NONE:
                continue

            feature_name = feature.name
            feature_index = Features.get_feature_index(feature, False)

            if feature.is_block_feature:
                raw_value = block_data[feature_name][0, x]
            else:
                raw_value = predictions[0, x, feature_index]

            if feature.is_block_feature:
                if event_type not in [EventType.BLOCK_ENTER.value, EventType.BLOCK_EXIT.value]:
                    continue
                if feature.encoding == EncodingType.TOKENIZED:
                    token = int(raw_value)
                    event[feature_name] = untokenizers[feature_name].get(token, f"UNKNOWN_{feature_name}_{token}")
                elif feature.encoding == EncodingType.ONE_HOT:
                    label = int(np.argmax(raw_value))
                    event[feature_name] = label
                else:
                    event[feature_name] = convert_value(raw_value)
            else:
                if feature.encoding == EncodingType.ONE_HOT:
                    label = int(np.argmax(raw_value))
                    event[feature_name] = label
                elif feature.encoding == EncodingType.TOKENIZED:
                    token = int(raw_value)
                    event[feature_name] = untokenizers[feature_name].get(token, f"UNKNOWN_{feature_name}_{token}")
                else:
                    event[feature_name] = convert_value(raw_value)

        events.append(event)

    # Write the events list to a JSON file
    with open(filename, 'w') as f:
        json.dump(events, f, indent=4)

    print(f"Saved {len(events)} events to {filename}")

def main():
    config = ModelConfig(deterministic=True)
    model = BasicTrackmaniaNN(config=config)
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate=0.001, steps_per_epoch=100)

    manager = TrackmaniaDataManager('trackmania_dataset.h5')
    map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'
    replay_name = 'replay_1'

    # Load global_stats from training
    global_stats = manager.load_global_stats(map_uid)

    # Retrieve the first 32 events and their block data to start prediction
    raw_events = manager.get_first_x_events_with_block_data(map_uid, replay_name, x=32)  # Start with 32 events
    inputs = np.expand_dims(raw_events, axis=0)

    # Preprocess the raw events using DataProcessor
    data_processor = DataProcessor(manager, map_uid, config, global_stats=global_stats)
    data_processor.load_all_tokenizers()
    data_processor.update_config_sizes()
    sequence = data_processor.preprocess_data(inputs)

    print(f"Sequence data shape: {sequence['data'].shape}")
    for key, value in sequence['blocks'].items():
        print(f"Block '{key}' shape: {value.shape}")

    batch_size = 1
    sequence_length = 32
    for feature in Features.get_block_features():
        if feature.encoding == EncodingType.TOKENIZED:
            expected_shape = (batch_size, sequence_length)
            actual_shape = sequence['blocks'][feature.name].shape
            assert actual_shape == expected_shape, f"Shape mismatch for {feature.name}: expected {expected_shape}, got {actual_shape}"
        elif feature.encoding == EncodingType.ONE_HOT:
            expected_shape = (batch_size, sequence_length, feature.size)
            actual_shape = sequence['blocks'][feature.name].shape
            assert actual_shape == expected_shape, f"Shape mismatch for {feature.name}: expected {expected_shape}, got {actual_shape}"


    # For debugging, print out the "Time" feature for each element in the sequence

    # Expand dimensions to simulate batch_size=1
    # sequence = {
    #     'data': np.expand_dims(preprocessed_sequence['data'], axis=0),      # Shape: (1, seq_length, num_features)
    #     'blocks': {key: np.expand_dims(value, axis=0) for key, value in preprocessed_sequence['blocks'].items()}  # Shape: (1, seq_length, ...)
    # }

    # Infer the shapes for inputs and block data
    input_shape = sequence['data'].shape  # Should be (1, seq_length, num_features)
    block_shapes = {key: value.shape for key, value in sequence['blocks'].items()}

    # Restore the model's state from a checkpoint
    state = create_train_state(rngs, model, learning_rate=learning_rate_fn, input_shape=input_shape, block_shapes=block_shapes)
    _, state = restore_train_state(state)

    # Perform autoregressive prediction
    predictions = autoregressive_predict(
        model=model, 
        initial_input=sequence['data'], 
        params=state.params, 
        block_data=sequence['blocks'], 
        output_length=5, 
        train=False,
        rngs=rngs
    )

    save_predictions_as_json(
        predictions=predictions['events'],
        block_data=predictions['blocks'],
        filename='predictions.json',
        data_processor=data_processor
    )
    

if __name__ == '__main__':
    main()
