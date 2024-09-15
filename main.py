import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Sequence
import numpy as np
import logging

from data_manager import TrackmaniaDataManager, EventType, tokenize_block

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_event_with_block_data(event, tokenized_blocks, global_position_mean, global_position_std):
    block = tokenized_blocks.get(event['BlockHash'].decode("utf-8"))
    if block is not None:
        block['Position'] = normalize_block_position(block['Position'][()], global_position_mean, global_position_std)
        return block
    return None

def preprocess_data(data, tokenized_blocks, global_stats):
    # Extract numerical fields using vectorized operations
    times = data['Time']
    event_types = data['EventType'].astype(jnp.int32)  # Ensure integer type
    input_steer = data['InputSteer']
    input_gas = data['InputGasPedal']
    input_brake = data['InputBrakePedal']
    positions = data['Position']
    velocities = data['Velocity']

    # Prepare event data
    event_data = np.concatenate([
        times[..., None],
        event_types[..., None],
        input_steer[..., None],
        input_gas[..., None],
        input_brake[..., None],
        positions,
        velocities
    ], axis=-1)

    # Handle block data
    # Create masks for events with block data
    block_event_mask = np.isin(event_types, [EventType.BLOCK_ENTER, EventType.BLOCK_EXIT])

    # Initialize block data arrays with default values
    num_sequences, sequence_length = event_types.shape
    block_data_shape = (num_sequences, sequence_length)
    block_data = {
        'PageName': np.full(block_data_shape, -1, dtype=np.int32),
        'MaterialName': np.full(block_data_shape, -1, dtype=np.int32),
        'Name': np.full(block_data_shape, -1, dtype=np.int32),
        'Position': np.zeros((num_sequences, sequence_length, 3), dtype=np.float32),
        'Direction': np.full(block_data_shape, -1, dtype=np.int32)
    }

    # Get indices where block data is present
    block_indices = np.where(block_event_mask)

    # Extract block hashes for events with block data
    block_hashes = data['BlockHash'][block_indices]

    # Prepare block data using vectorized operations
    for field in block_data.keys():
        field_values = []
        for block_hash in block_hashes:
            block = tokenized_blocks.get(block_hash.decode("utf-8"))
            if block is not None:
                value = block[field]
            else:
                if field == 'Position':
                    value = np.zeros(3, dtype=np.float32)
                elif field == 'Direction':
                    value = 0
                elif field in ['PageName', 'MaterialName', 'Name']:
                    value = -1
                else:
                    raise ValueError(f"Unknown field {field}")
            field_values.append(value)
        field_values = np.array(field_values)
        block_data[field][block_indices] = field_values


    # Convert block data to JAX arrays
    for key in block_data:
        block_data[key] = jnp.array(block_data[key])

    result = {
        'inputs': jnp.array(event_data, dtype=jnp.float32),
        'blocks': block_data
    }
    return result

@jax.jit
def normalize_data(data, global_stats):
    # Extract features
    time = data[..., 0:1]
    event_type = data[..., 1].astype(jnp.int32)  # Ensure integer type
    inputs = data[..., 2:5]
    position = data[..., 5:8]
    velocity = data[..., 8:11]

    # Normalize numerical features
    time_normalized = time - jnp.min(time, axis=-2, keepdims=True)
    inputs_normalized = inputs
    position_normalized = (position - global_stats[0]) / (global_stats[1] + 1e-8)
    velocity_normalized = (velocity - global_stats[2]) / (global_stats[3] + 1e-8)

    # Concatenate normalized numerical features
    normalized_numerical = jnp.concatenate([
        time_normalized, inputs_normalized, position_normalized, velocity_normalized
    ], axis=-1)

    # Return a dictionary with numerical and categorical data
    normalized_data = {
        'numerical': normalized_numerical,
        'event_type': event_type  # Keep event_type separate
    }
    return normalized_data

@jax.jit
def normalize_block_position(block_position, global_mean, global_std):
    return (block_position - global_mean) / (global_std + 1e-8)

class BlockEmbedding(nn.Module):
    vocab_size: int
    embedding_dim: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)

    def __call__(self, x):
        return self.embedding(x)

class BasicTrackmaniaNN(nn.Module):
    hidden_sizes: Sequence[int]
    output_sizes: dict
    block_vocab_sizes: dict
    block_embedding_dim: int

    @nn.compact
    def __call__(self, x, block_data, event_type, train: bool = True):
        # Define block embeddings
        page_name_embedding = nn.Embed(num_embeddings=self.block_vocab_sizes['PageName'], features=self.block_embedding_dim)
        material_name_embedding = nn.Embed(num_embeddings=self.block_vocab_sizes['MaterialName'], features=self.block_embedding_dim)
        name_embedding = nn.Embed(num_embeddings=self.block_vocab_sizes['Name'], features=self.block_embedding_dim)

        # Convert block_data into arrays
        page_name = jnp.asarray(block_data['PageName'], dtype=jnp.int32)
        material_name = jnp.asarray(block_data['MaterialName'], dtype=jnp.int32)
        name = jnp.asarray(block_data['Name'], dtype=jnp.int32)
        position = block_data['Position']
        direction = block_data['Direction']

        # Get block embeddings
        page_name_embed = page_name_embedding(page_name)
        material_name_embed = material_name_embedding(material_name)
        name_embed = name_embedding(name)

        # Concatenate embeddings and block features
        block_embeddings = jnp.concatenate([page_name_embed, material_name_embed, name_embed], axis=-1)

        # Concatenate block embeddings and features with the input data
        block_features = jnp.concatenate([block_embeddings, position, direction[..., None]], axis=-1)
        x = jnp.concatenate([x, block_features, event_type[..., None]], axis=-1)

        # Output layers for each output
        outputs = {}

        # Existing outputs
        outputs['time'] = nn.Dense(1)(x)
        outputs['inputs'] = nn.Dense(3)(x)
        outputs['position'] = nn.Dense(3)(x)
        outputs['velocity'] = nn.Dense(3)(x)
        outputs['event_type'] = nn.Dense(self.output_sizes['event_type'])(x)

        # New outputs
        outputs['block_name'] = nn.Dense(self.output_sizes['block_name'])(x)
        outputs['block_position'] = nn.Dense(3)(x)
        outputs['block_direction'] = nn.Dense(self.output_sizes['block_direction'])(x)

        return outputs

def create_train_state(rng, model, learning_rate, input_shape, block_shapes):
    logger.debug(f"Creating train state with input shape: {input_shape} and block shape: {block_shapes}")
    
    batch_size = input_shape[0]
    dummy_input = jnp.ones(input_shape)
    dummy_event_type = jnp.ones((batch_size, input_shape[1]), dtype=jnp.int32)
    dummy_block_data = {
        'PageName': jnp.ones((batch_size,) + block_shapes['PageName'], dtype=jnp.int32),
        'MaterialName': jnp.ones((batch_size,) + block_shapes['MaterialName'], dtype=jnp.int32),
        'Name': jnp.ones((batch_size,) + block_shapes['Name'], dtype=jnp.int32),
        'Position': jnp.ones((batch_size,) + block_shapes['Position'], dtype=jnp.float32),
        'Direction': jnp.ones((batch_size,) + block_shapes['Direction'], dtype=jnp.int32)
    }

    params = model.init(rng, dummy_input, dummy_block_data, dummy_event_type)['params']
    
    tx = optax.chain(optax.adam(learning_rate), optax.clip_by_global_norm(1.0))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def custom_loss(predictions, targets, loss_weights):
    # Extract predictions
    pred_time = predictions['time']
    pred_inputs = predictions['inputs']
    pred_position = predictions['position']
    pred_velocity = predictions['velocity']
    pred_event_type = predictions['event_type']

    # New predictions
    pred_block_name = predictions['block_name']
    pred_block_position = predictions['block_position']
    pred_block_direction = predictions['block_direction']

    # Extract targets
    true_time = targets['time']
    true_inputs = targets['inputs']
    true_position = targets['position']
    true_velocity = targets['velocity']
    true_event_type = targets['event_type'].astype(jnp.int32)

    # New targets
    true_block_name = targets['block_name']
    true_block_position = targets['block_position']
    true_block_direction = targets['block_direction']

    # Create mask for events with block data
    block_event_mask = jnp.isin(true_event_type, jnp.array([EventType.BLOCK_ENTER, EventType.BLOCK_EXIT]))
    # block_event_mask = jnp.squeeze(block_event_mask, axis=-1)

    # Compute losses
    inputs_loss = jnp.mean((pred_inputs - true_inputs) ** 2)
    time_loss = jnp.mean((pred_time - true_time) ** 2)
    position_loss = jnp.mean((pred_position - true_position) ** 2)
    velocity_loss = jnp.mean((pred_velocity - true_velocity) ** 2)
    # event_type_loss = jnp.mean(optax.softmax_cross_entropy(pred_event_type, true_event_type))

    # Compute block-related losses using the mask
    block_name_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(pred_block_name, true_block_name) * block_event_mask
    )

    block_position_loss = jnp.mean(
        jnp.sum((pred_block_position - true_block_position) ** 2, axis=-1) * block_event_mask
    )

    event_type_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(pred_event_type, true_event_type)
    )

    block_direction_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(pred_block_direction, true_block_direction) * block_event_mask
    )


    # Total loss
    total_loss = (
        loss_weights['position'] * position_loss +
        loss_weights['velocity'] * velocity_loss +
        loss_weights['event_type'] * event_type_loss +
        inputs_loss +
        time_loss * loss_weights['time'] +
        loss_weights['block_name'] * block_name_loss +
        loss_weights['block_position'] * block_position_loss +
        loss_weights['block_direction'] * block_direction_loss
    )

    return total_loss


@jax.jit
def train_step(state, batch, loss_weights):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['inputs'], batch['blocks'], batch['event_type'])
        return custom_loss(predictions, batch['targets'], loss_weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, batch, loss_weights):
    predictions = state.apply_fn({'params': state.params}, batch['inputs'], batch['blocks'], batch['event_type'])
    return custom_loss(predictions, batch['targets'], loss_weights)

@jax.jit
def compute_event_type_accuracy(predictions, targets):
    pred_event_type = predictions[..., 10:]
    true_event_type = targets[..., 10:]
    
    pred_class = jnp.argmax(pred_event_type, axis=-1)
    true_class = jnp.argmax(true_event_type, axis=-1)
    
    accuracy = jnp.mean(pred_class == true_class)
    
    return accuracy

@jax.jit
def compute_position_velocity_accuracy(predictions, targets, position_tolerance=1.0, velocity_tolerance=1.0):
    pred_position = predictions[..., 4:7]
    true_position = targets[..., 4:7]
    
    pred_velocity = predictions[..., 7:10]
    true_velocity = targets[..., 7:10]
    
    position_error = jnp.abs(pred_position - true_position)
    velocity_error = jnp.abs(pred_velocity - true_velocity)
    
    position_accuracy = jnp.mean(jnp.all(position_error < position_tolerance, axis=-1))
    velocity_accuracy = jnp.mean(jnp.all(velocity_error < velocity_tolerance, axis=-1))
    
    return position_accuracy, velocity_accuracy

@jax.jit
def compute_test_accuracy(predictions, targets, position_tolerance=1.0, velocity_tolerance=1.0):
    event_type_accuracy = compute_event_type_accuracy(predictions, targets)
    
    position_accuracy, velocity_accuracy = compute_position_velocity_accuracy(predictions, targets, position_tolerance, velocity_tolerance)
    
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
        
        predictions = state.apply_fn({'params': state.params}, batch['inputs'], batch['blocks'])
        
        batch_accuracy = compute_test_accuracy(predictions, batch['targets'], position_tolerance, velocity_tolerance)
        accuracies.append(batch_accuracy)
    
    mean_accuracies = {
        'event_type_accuracy': jnp.mean(jnp.array([acc['event_type_accuracy'] for acc in accuracies])),
        'position_accuracy': jnp.mean(jnp.array([acc['position_accuracy'] for acc in accuracies])),
        'velocity_accuracy': jnp.mean(jnp.array([acc['velocity_accuracy'] for acc in accuracies]))
    }
    
    return mean_accuracies

def create_batch(train_data, batch_size):
    inputs = train_data['inputs']
    event_type = train_data['event_type']
    block_data = train_data['blocks']
    targets = train_data['targets']

    num_samples = len(inputs)
    for i in range(0, num_samples, batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_event_type = event_type[i:i+batch_size]  # Shape: (batch_size, seq_length)
        batch_block_data = {key: block_data[key][i:i+batch_size] for key in block_data}

        batch_targets = {
            'time': targets['numerical'][i:i+batch_size, :, 0:1],
            'inputs': targets['numerical'][i:i+batch_size, :, 1:4],
            'position': targets['numerical'][i:i+batch_size, :, 4:7],
            'velocity': targets['numerical'][i:i+batch_size, :, 7:10],
            'event_type': targets['event_type'][i:i+batch_size],
            'block_name': targets['blocks']['Name'][i:i+batch_size],
            'block_position': targets['blocks']['Position'][i:i+batch_size],
            'block_direction': targets['blocks']['Direction'][i:i+batch_size]
        }

        yield {
            'inputs': batch_inputs,
            'event_type': batch_event_type,
            'blocks': batch_block_data,
            'targets': batch_targets
        }

def train_model(train_data, test_data, model, num_epochs, batch_size, learning_rate):
    rng = jax.random.PRNGKey(0)
    input_shape = train_data['inputs'].shape[1:]
    logger.debug(f"Train model input shape: {input_shape}")
    block_shapes = {
        'PageName': train_data['blocks']['PageName'].shape[1:],  # Shape excluding batch size
        'MaterialName': train_data['blocks']['MaterialName'].shape[1:],
        'Name': train_data['blocks']['Name'].shape[1:],
        'Position': train_data['blocks']['Position'].shape[1:],  # Shape for 3D position
        'Direction': train_data['blocks']['Direction'].shape[1:]  # Shape for direction
    }
    state = create_train_state(rng, model, learning_rate, (1,) + input_shape, block_shapes)

    loss_weights = {
        'position': 0.01,
        'velocity': 0.01,
        'event_type': 1,
        'time': 0.00001,
        'block_name': 1.0,
        'block_position': 0.01,
        'block_direction': 1.0
    }


    for epoch in range(num_epochs):
        # Training
        batch_losses = []
        for batch in create_batch(train_data, batch_size):
            state, loss = train_step(state, batch, loss_weights)
            batch_losses.append(loss)
        train_loss = jnp.mean(jnp.array(batch_losses))

        # Evaluation
        batch_losses = []
        for batch in create_batch(test_data, batch_size):
            loss = eval_step(state, batch, loss_weights)
            batch_losses.append(loss)
        test_loss = jnp.mean(jnp.array(batch_losses))

        # accuracies = eval_model_accuracy(state, test_data, batch_size)
        # logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} Event type accuracy: {accuracies['event_type_accuracy']:.4f}, Position accuracy: {accuracies['position_accuracy']:.4f}, Velocity accuracy: {accuracies['velocity_accuracy']:.4f}")
        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return state

def extract_block_hashes(sequences):
    block_hashes = set()
    for sequence in sequences:
        for event in sequence:
            if event['EventType'] in (EventType.BLOCK_ENTER, EventType.BLOCK_EXIT):
                block_hash = event['BlockHash'].decode('utf-8')
                block_hashes.add(block_hash)
    return block_hashes


def prepare_data(manager, map_uid):
    raw_data = manager.prepare_data_for_training(map_uid, sequence_length=10, stride=1, test_split=0.2)

    # Extract block hashes from training inputs and targets
    train_block_hashes = extract_block_hashes(raw_data['train_inputs'])
    test_block_hashes = extract_block_hashes(raw_data['test_inputs'])

    # Combine block hashes from training and testing data
    all_block_hashes = train_block_hashes.union(test_block_hashes)

    # Pass the set of block hashes to get_tokenizers
    tokenizers, tokenized_blocks = manager.get_tokenizers(map_uid, all_block_hashes)

    if 'global_stats' in manager.file[f'maps/{map_uid}']:
        print(f"Loading global stats from HDF5 for {map_uid}")
        global_stats = manager.load_global_stats(map_uid)
    else:
        train_numeric_data = extract_numeric_data(raw_data['train_inputs'])
        global_stats = calculate_global_stats(train_numeric_data)

        manager.save_global_stats(map_uid, global_stats)


    train_numeric_data = extract_numeric_data(raw_data['train_inputs'])
    global_stats = calculate_global_stats(train_numeric_data)


    train_processed = preprocess_data(raw_data['train_inputs'], tokenized_blocks, global_stats)
    train_inputs = train_processed['inputs']
    train_block_data = train_processed['blocks']

    train_targets_processed = preprocess_data(raw_data['train_targets'], tokenized_blocks, global_stats)
    train_targets = {
        'inputs': train_targets_processed['inputs'],
        'blocks': train_targets_processed['blocks']
    }

    test_processed = preprocess_data(raw_data['test_inputs'], tokenized_blocks, global_stats)
    test_inputs = test_processed['inputs']
    test_block_data = test_processed['blocks']

    test_targets_processed = preprocess_data(raw_data['test_targets'], tokenized_blocks, global_stats)
    test_targets = {
        'inputs': test_targets_processed['inputs'],
        'blocks': test_targets_processed['blocks']
    }

    # Normalize data
    train_inputs_normalized = normalize_data(train_inputs, global_stats)
    train_inputs = train_inputs_normalized['numerical']
    train_event_type = train_inputs_normalized['event_type']

    train_targets_normalized = normalize_data(train_targets['inputs'], global_stats)
    train_targets['numerical'] = train_targets_normalized['numerical']
    train_targets['event_type'] = train_targets_normalized['event_type']

    # Similarly for test data
    test_inputs_normalized = normalize_data(test_inputs, global_stats)
    test_inputs = test_inputs_normalized['numerical']
    test_event_type = test_inputs_normalized['event_type']

    test_targets_normalized = normalize_data(test_targets['inputs'], global_stats)
    test_targets['numerical'] = test_targets_normalized['numerical']
    test_targets['event_type'] = test_targets_normalized['event_type']

    # Update train_data and test_data dictionaries
    train_data = {
        'inputs': train_inputs,
        'event_type': train_event_type,
        'blocks': train_block_data,
        'targets': train_targets
    }
    test_data = {
        'inputs': test_inputs,
        'event_type': test_event_type,
        'blocks': test_block_data,
        'targets': test_targets
    }
    return {
        'train': train_data,
        'test': test_data,
        'normalize': global_stats,
        'tokenizers': tokenizers
    }

def extract_numeric_data(inputs):
    # Assuming inputs is a structured NumPy array
    times = inputs['Time']  # Shape: (num_sequences, sequence_length)
    input_steer = inputs['InputSteer']
    input_gas = inputs['InputGasPedal']
    input_brake = inputs['InputBrakePedal']
    positions = inputs['Position']  # Shape: (num_sequences, sequence_length, 3)
    velocities = inputs['Velocity']

    # Concatenate all features along the last axis
    numeric_data = np.concatenate([
        times[..., None],
        input_steer[..., None],
        input_gas[..., None],
        input_brake[..., None],
        positions,
        velocities
    ], axis=-1)

    return jnp.array(numeric_data, dtype=jnp.float32)


def calculate_global_stats(train_inputs):    
    position_data = train_inputs[..., 4:7].reshape(-1, 3)
    velocity_data = train_inputs[..., 7:10].reshape(-1, 3)
    
    # Calculate global mean and standard deviation
    global_position_mean = jnp.mean(position_data, axis=0)
    global_position_std = jnp.std(position_data, axis=0)
    
    global_velocity_mean = jnp.mean(velocity_data, axis=0)
    global_velocity_std = jnp.std(velocity_data, axis=0)
    
    return global_position_mean, global_position_std, global_velocity_mean, global_velocity_std


manager = TrackmaniaDataManager('trackmania_dataset.h5')
map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'

data = prepare_data(manager, map_uid)

model = BasicTrackmaniaNN(
    hidden_sizes=[64, 32],
    output_sizes={
        'event_type': len(EventType),
        'block_name': len(data['tokenizers']['Name']),
        'block_direction': 4
    },
    block_vocab_sizes={
        'Name': len(data['tokenizers']['Name']),
        'PageName': len(data['tokenizers']['PageName']),
        'MaterialName': len(data['tokenizers']['MaterialName'])
    },
    block_embedding_dim=16
)

trained_state = train_model(
    data['train'],
    data['test'],
    model,
    num_epochs=1000,
    batch_size=32,
    learning_rate=0.01
)

@jax.jit
def generate_event(state, input_event, global_stats):
    global_position_mean, global_position_std, global_velocity_mean, global_velocity_std = global_stats
    
    normalized_input_event = normalize_data(jnp.array(input_event), global_stats)
    prediction_normalized = state.apply_fn({'params': state.params}, normalized_input_event[jnp.newaxis, :])

    predicted_position_normalized = prediction_normalized[0, 4:7]
    predicted_velocity_normalized = prediction_normalized[0, 7:10]

    predicted_position = predicted_position_normalized * global_position_std + global_position_mean
    predicted_velocity = predicted_velocity_normalized * global_velocity_std + global_velocity_mean

    generated_event = jnp.concatenate([
        prediction_normalized[0, :4],  # Time and inputs
        predicted_position,
        predicted_velocity,
        prediction_normalized[0, 10:]  # Event type
    ])

    return generated_event

initial_event = data['test']['inputs'][0]
generated_event = generate_event(trained_state, initial_event, data['normalize']['position_mean'], data['normalize']['position_std'], data['normalize']['velocity_mean'], data['normalize']['velocity_std'])

logger.debug(f"Generated event: {generated_event}")