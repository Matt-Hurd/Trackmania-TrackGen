import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Sequence
import numpy as np
import logging

from data_manager import TrackmaniaDataManager, EventType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(data, is_sequence=True):
    def process_event(event):
        event_data = [
            event['Time'],
            event['InputSteer'],
            event['InputGasPedal'],
            event['InputBrakePedal']
        ]

        for varkey in ['Position', 'Velocity']:
            for n in range(3):
                event_data.append(event[varkey][n])
                
        event_type_int = int(EventType(event['EventType']))
        event_type_one_hot = np.zeros(len(EventType))
        event_type_one_hot[event_type_int] = 1
        event_data.extend(event_type_one_hot)

        return event_data
    
    if is_sequence:
        processed_data = []
        for sequence in data:
            sequence_data = [process_event(event) for event in sequence]
            processed_data.append(sequence_data)
    else:
        processed_data = [process_event(event) for event in data]

    result = jnp.array(processed_data, dtype=jnp.float32)
    logger.debug(f"Preprocessed data shape: {result.shape}, dtype: {result.dtype}")
    return result

def normalize_data(data, global_position_mean, global_position_std, global_velocity_mean, global_velocity_std):
    # Separate different types of data
    time = data[..., 0:1]
    inputs = data[..., 1:4]  # InputSteer, InputGasPedal, InputBrakePedal
    position = data[..., 4:7]
    velocity = data[..., 7:10]
    event_type = data[..., 10:]  # One-hot encoded, no normalization needed

    
    # Normalize time relative to sequence start
    time_normalized = time - jnp.min(time, axis=-2, keepdims=True)
    
    # Inputs are assumed to be in a fixed range, no normalization needed
    inputs_normalized = inputs


    # Normalize position relative to the global mean and std
    position_normalized = (position - global_position_mean) / (global_position_std + 1e-8)

    # Normalize velocity similarly
    velocity_normalized = (velocity - global_velocity_mean) / (global_velocity_std + 1e-8)

    
    # Debug print out min/max values
    logger.debug(f"time_normalized min: {jnp.min(time_normalized)}, max: {jnp.max(time_normalized)}")
    logger.debug(f"inputs_normalized min: {jnp.min(inputs_normalized)}, max: {jnp.max(inputs_normalized)}")
    logger.debug(f"position_normalized min: {jnp.min(position_normalized)}, max: {jnp.max(position_normalized)}")
    logger.debug(f"velocity_normalized min: {jnp.min(velocity_normalized)}, max: {jnp.max(velocity_normalized)}")
    
    # Combine normalized data
    normalized_data = jnp.concatenate([
        time_normalized, inputs_normalized, position_normalized, 
        velocity_normalized, event_type
    ], axis=-1)
    
    return normalized_data

class BasicTrackmaniaNN(nn.Module):
    hidden_sizes: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        logger.debug(f"Model input shape: {x.shape}, dtype: {x.dtype}")
        x = x.reshape((x.shape[0], -1))
        logger.debug(f"Flattened input shape: {x.shape}")
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size, kernel_init=nn.initializers.glorot_uniform())(x)
            logger.debug(f"After Dense layer {i+1} shape: {x.shape}")
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.output_size, kernel_init=nn.initializers.glorot_uniform())(x)
        logger.debug(f"Model output shape: {x.shape}")
        return x

def create_train_state(rng, model, learning_rate, input_shape):
    logger.debug(f"Creating train state with input shape: {input_shape}")
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.chain(optax.adam(learning_rate), optax.clip_by_global_norm(1.0))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def custom_loss(predictions, targets, weights):
    pred_time = predictions[..., 0:1]          # Time
    pred_inputs = predictions[..., 1:4]        # Steer, Gas, Brake inputs
    pred_position = predictions[..., 4:7]      # Position (X, Y, Z)
    pred_velocity = predictions[..., 7:10]     # Velocity (X, Y, Z)
    pred_event_type = predictions[..., 10:]    # Event type

    true_time = targets[..., 0:1]
    true_inputs = targets[..., 1:4]
    true_position = targets[..., 4:7]
    true_velocity = targets[..., 7:10]
    true_event_type = targets[..., 10:]

    inputs_loss = jnp.mean((pred_inputs - true_inputs) ** 2)
    time_loss = jnp.mean((pred_time - true_time) ** 2)

    # np_pred_time = jnp.array(pred_time)
    # np_true_time = jnp.array(true_time)

    # print(f"Predicted time: {np_pred_time}, True time: {np_true_time}")
    
    # Compute individual losses
    position_loss = jnp.mean((pred_position - true_position) ** 2)
    velocity_loss = jnp.mean((pred_velocity - true_velocity) ** 2)
    event_type_loss = jnp.mean(optax.softmax_cross_entropy(pred_event_type, true_event_type))

    # Combine losses with weights
    total_loss = (
        weights['position'] * position_loss +
        weights['velocity'] * velocity_loss +
        weights['event_type'] * event_type_loss + 
        inputs_loss + 
        time_loss * 0.00001
    )

    # Clip total loss to avoid extreme values
    # total_loss = jnp.clip(total_loss, a_min=-100, a_max=100)
    return total_loss


@jax.jit
def train_step(state, batch, loss_weights):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['inputs'])
        return custom_loss(predictions, batch['targets'], loss_weights)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # logger.debug(f"Gradient norms: {jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grads)}")
    # return jnp.mean(optax.l2_loss(predictions, batch['targets']))
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, batch, loss_weights):
    predictions = state.apply_fn({'params': state.params}, batch['inputs'])
    return custom_loss(predictions, batch['targets'], loss_weights)

def compute_event_type_accuracy(predictions, targets):
    # Get the predicted and true event types (as one-hot vectors)
    pred_event_type = predictions[..., 10:]
    true_event_type = targets[..., 10:]
    
    # Get the index of the maximum value (the predicted class) for both prediction and target
    pred_class = jnp.argmax(pred_event_type, axis=-1)
    true_class = jnp.argmax(true_event_type, axis=-1)
    
    # Compute accuracy: how many predictions match the true classes
    accuracy = jnp.mean(pred_class == true_class)
    
    return accuracy

def compute_position_velocity_accuracy(predictions, targets, position_tolerance=1.0, velocity_tolerance=1.0):
    # Extract position and velocity predictions and targets
    pred_position = predictions[..., 4:7]
    true_position = targets[..., 4:7]
    
    pred_velocity = predictions[..., 7:10]
    true_velocity = targets[..., 7:10]
    
    # Compute absolute errors
    position_error = jnp.abs(pred_position - true_position)
    velocity_error = jnp.abs(pred_velocity - true_velocity)
    
    # Check if errors are within the tolerance
    position_accuracy = jnp.mean(jnp.all(position_error < position_tolerance, axis=-1))
    velocity_accuracy = jnp.mean(jnp.all(velocity_error < velocity_tolerance, axis=-1))
    
    return position_accuracy, velocity_accuracy


def compute_test_accuracy(predictions, targets, position_tolerance=1.0, velocity_tolerance=1.0):
    # Compute accuracy for event type (classification)
    event_type_accuracy = compute_event_type_accuracy(predictions, targets)
    
    # Compute accuracy for position and velocity (regression)
    position_accuracy, velocity_accuracy = compute_position_velocity_accuracy(predictions, targets, position_tolerance, velocity_tolerance)
    
    # Optionally, you can combine them or report them separately
    overall_accuracy = {
        'event_type_accuracy': event_type_accuracy,
        'position_accuracy': position_accuracy,
        'velocity_accuracy': velocity_accuracy
    }
    
    return overall_accuracy

def eval_model_accuracy(state, test_data, batch_size, position_tolerance=1.0, velocity_tolerance=1.0):
    accuracies = []
    
    for i in range(0, len(test_data['inputs']), batch_size):
        batch = {
            'inputs': test_data['inputs'][i:i+batch_size],
            'targets': test_data['targets'][i:i+batch_size]
        }
        
        # Get the model's predictions
        predictions = state.apply_fn({'params': state.params}, batch['inputs'])
        
        # Compute accuracy for this batch
        batch_accuracy = compute_test_accuracy(predictions, batch['targets'], position_tolerance, velocity_tolerance)
        accuracies.append(batch_accuracy)
    
    # Average the accuracies across batches
    mean_accuracies = {
        'event_type_accuracy': jnp.mean(jnp.array([acc['event_type_accuracy'] for acc in accuracies])),
        'position_accuracy': jnp.mean(jnp.array([acc['position_accuracy'] for acc in accuracies])),
        'velocity_accuracy': jnp.mean(jnp.array([acc['velocity_accuracy'] for acc in accuracies]))
    }
    
    return mean_accuracies


def train_model(train_data, test_data, model, num_epochs, batch_size, learning_rate):
    rng = jax.random.PRNGKey(0)
    input_shape = train_data['inputs'].shape[1:]
    logger.debug(f"Train model input shape: {input_shape}")
    state = create_train_state(rng, model, learning_rate, (1,) + input_shape)

    loss_weights = {
        'position': 0.01,
        'velocity': 0.01,
        'event_type': 1.0
    }

    for epoch in range(num_epochs):
        # Training
        batch_losses = []
        for i in range(0, len(train_data['inputs']), batch_size):
            batch = {
                'inputs': train_data['inputs'][i:i+batch_size],
                'targets': train_data['targets'][i:i+batch_size]
            }
            logger.debug(f"Batch inputs shape: {batch['inputs'].shape}, targets shape: {batch['targets'].shape}")
            state, loss = train_step(state, batch, loss_weights)
            batch_losses.append(loss)
        train_loss = jnp.mean(jnp.array(batch_losses))

        # Evaluation
        batch_losses = []
        for i in range(0, len(test_data['inputs']), batch_size):
            batch = {
                'inputs': test_data['inputs'][i:i+batch_size],
                'targets': test_data['targets'][i:i+batch_size]
            }
            loss = eval_step(state, batch, loss_weights)
            batch_losses.append(loss)
        test_loss = jnp.mean(jnp.array(batch_losses))

        accuracies = eval_model_accuracy(state, test_data, batch_size)
        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} Event type accuracy: {accuracies['event_type_accuracy']:.4f}, Position accuracy: {accuracies['position_accuracy']:.4f}, Velocity accuracy: {accuracies['velocity_accuracy']:.4f}")

    return state

def prepare_data(manager, map_uid):
    raw_data = manager.prepare_data_for_training(map_uid, sequence_length=10, stride=1, test_split=0.2)
    
    train_inputs = preprocess_data(raw_data['train_inputs'], is_sequence=True)
    train_targets = preprocess_data(raw_data['train_targets'], is_sequence=False)
    
    test_inputs = preprocess_data(raw_data['test_inputs'], is_sequence=True)
    test_targets = preprocess_data(raw_data['test_targets'], is_sequence=False)

    global_position_mean, global_position_std, global_velocity_mean, global_velocity_std = calculate_global_stats(train_inputs)
    
    # Normalize the data
    train_inputs = normalize_data(train_inputs, global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)
    train_targets = normalize_data(train_targets, global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)
    
    test_inputs = normalize_data(test_inputs, global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)
    test_targets = normalize_data(test_targets, global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)
    
    return {
        'train': {
            'inputs': train_inputs,
            'targets': train_targets
        },
        'test': {
            'inputs': test_inputs,
            'targets': test_targets
        },
        'normalize': {
            'position_mean': global_position_mean,
            'position_std': global_position_std,
            'velocity_mean': global_velocity_mean,
            'velocity_std': global_velocity_std
        }
    }

def calculate_global_stats(train_inputs):
    # Assuming that position is at index 4:7 and velocity is at index 7:10 in the inputs
    # Flatten the data to (N, 3) where N is the total number of samples across all sequences
    
    position_data = train_inputs[..., 4:7].reshape(-1, 3)  # (total_samples, 3)
    velocity_data = train_inputs[..., 7:10].reshape(-1, 3)  # (total_samples, 3)
    
    # Calculate global mean and standard deviation
    global_position_mean = jnp.mean(position_data, axis=0)
    global_position_std = jnp.std(position_data, axis=0)
    
    global_velocity_mean = jnp.mean(velocity_data, axis=0)
    global_velocity_std = jnp.std(velocity_data, axis=0)
    
    return global_position_mean, global_position_std, global_velocity_mean, global_velocity_std


manager = TrackmaniaDataManager('trackmania_dataset.h5')
map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'

data = prepare_data(manager, map_uid)

logger.debug(f"Final input shape: {data['train']['inputs'].shape}")
logger.debug(f"Final output shape: {data['train']['targets'].shape}")

input_size = np.prod(data['train']['inputs'].shape[1:])
output_size = data['train']['targets'].shape[1]

logger.debug(f"Input size: {input_size}, Output size: {output_size}")

model = BasicTrackmaniaNN(
    hidden_sizes=[64, 32],
    output_size=output_size
)

trained_state = train_model(
    data['train'],
    data['test'],
    model,
    num_epochs=1000,
    batch_size=32,
    learning_rate=0.01
)

# Function to generate new events
def generate_event(state, input_event, global_position_mean, global_position_std, global_velocity_mean, global_velocity_std):
    normalized_input_event = normalize_data(jnp.array(input_event), global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)
    # Generate the prediction (note that input needs to be batched with jnp.newaxis)
    prediction_normalized = state.apply_fn({'params': state.params}, normalized_input_event[jnp.newaxis, :])

    # Denormalize the predicted position and velocity
    predicted_position_normalized = prediction_normalized[0, 4:7]
    predicted_velocity_normalized = prediction_normalized[0, 7:10]

    predicted_position = predicted_position_normalized * global_position_std + global_position_mean
    predicted_velocity = predicted_velocity_normalized * global_velocity_std + global_velocity_mean

    # Reconstruct the full event with the denormalized position and velocity
    generated_event = jnp.concatenate([
        prediction_normalized[0, :4],  # Time and inputs
        predicted_position,
        predicted_velocity,
        prediction_normalized[0, 10:]  # Event type
    ])

    return generated_event

# Generate a new event
initial_event = data['test']['inputs'][0]
generated_event = generate_event(trained_state, initial_event, data['normalize']['position_mean'], data['normalize']['position_std'], data['normalize']['velocity_mean'], data['normalize']['velocity_std'])

logger.debug(f"Generated event: {generated_event}")