from typing import List

import jax
import numpy as np
import jax.numpy as jnp

from data_manager import TrackmaniaDataManager
from enums import EncodingType
from features import Features
from main import CHECKPOINT_DIR, BasicTrackmaniaNN, DataProcessor, ModelConfig, create_learning_rate_fn, get_block_data_for_hashes, restore_train_state

def autoregressive_predict(model, initial_input, params, block_data=None, target_length=100, train=False, rngs=None):
    """
    Perform autoregressive prediction. Starting with the initial input, predict one step at a time,
    feeding the prediction back into the model as input for the next step.
    """
    # Initialize the input sequence with the given initial input
    predicted_sequence = initial_input
    current_input = initial_input  # Start with the initial input

    # Loop until the target sequence length is reached
    for _ in range(target_length - initial_input.shape[1]):  # Subtract initial length to predict remaining steps
        # Predict the next step based on the current input
        next_step = model.apply({'params': params}, current_input, block_data=block_data, train=train, rngs=rngs)
        
        next_step_prediction = []
        # Get the last predicted step (assuming it's the last time step in the output)
        for feature in Features:
            if feature.encoding == EncodingType.NONE:
                continue
            print(f"{feature.name} shape", next_step[feature.name].shape)
            pred = next_step[feature.name][:, -1:, :]
            print(f"{feature.name} pred shape", pred.shape)
            if feature.is_block_feature:
                block_data[feature.name] = jnp.concatenate([block_data[feature.name], pred], axis=1)
            else:
                next_step_prediction.append(pred)

        next_step_prediction = jnp.concatenate(next_step_prediction, axis=-1)

        predicted_sequence = jnp.concatenate([predicted_sequence, next_step_prediction], axis=1)
        
        # Use the new sequence as the input for the next prediction step
        current_input = predicted_sequence

    # Return the complete predicted sequence
    return predicted_sequence


def main():
    config = ModelConfig()
    model = BasicTrackmaniaNN(config=config)
    rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate=0.001, steps_per_epoch=100)

    manager = TrackmaniaDataManager('trackmania_dataset.h5')
    map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'
    replay_name = 'replay_1'

    # Retrieve the first 10 events and their block data to start prediction
    inputs = manager.get_first_x_events_with_block_data(map_uid, replay_name, x=10)  # Start with 10 events
    inputs = np.expand_dims(inputs, axis=0)

    # Process data (preprocessing, normalization, etc.)
    data_processor = DataProcessor(manager, map_uid, config)
    data_processor.prepare_data()
    sequence = data_processor.preprocess_data(inputs)

    # Infer the shapes for inputs and block data
    input_shape = sequence['data'].shape  # Use the actual input sequence shape
    block_shapes = {key: value.shape for key, value in sequence['blocks'].items()}

    # Restore or initialize the model's state from a checkpoint
    state = restore_train_state(
        checkpoint_dir=CHECKPOINT_DIR,
        model=model,
        learning_rate=learning_rate_fn,
        input_shape=input_shape,  # Dynamic input shape
        block_shapes=block_shapes,
        rngs=rngs
    )

    # Define the target length for the predicted sequence (e.g., 100 steps)
    target_length = 100

    # Perform autoregressive prediction
    predictions = autoregressive_predict(
        model=model, 
        initial_input=sequence['data'], 
        params=state.params, 
        block_data=sequence['blocks'], 
        target_length=target_length, 
        train=False,
        rngs=rngs
    )

    # Print or return predictions
    print(f"Predicted sequence length: {predictions.shape[1]}")
    print("Predictions:", predictions)


if __name__ == '__main__':
    main()
