import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Tuple, Dict, Any
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum

from data_manager import TrackmaniaDataManager, EventType, tokenize_block

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define enums for input and output keys
class InputKeys(Enum):
    TIME = 'Time'
    EVENT_TYPE = 'EventType'
    INPUT_STEER = 'InputSteer'
    INPUT_GAS = 'InputGasPedal'
    INPUT_BRAKE = 'InputBrakePedal'
    POSITION = 'Position'
    VELOCITY = 'Velocity'
    BLOCK_HASH = 'BlockHash'

class OutputKeys(Enum):
    TIME = 'time'
    INPUTS = 'inputs'
    POSITION = 'position'
    VELOCITY = 'velocity'
    EVENT_TYPE = 'event_type'
    BLOCK_NAME = 'block_name'
    BLOCK_POSITION = 'block_position'
    BLOCK_DIRECTION = 'block_direction'
    PAGE_NAME = 'page_name'

# Centralized configuration for inputs and outputs
@dataclass
class ModelConfig:
    input_features: Dict[str, Any] = field(default_factory=lambda: {
        'numerical': [
            {'key': InputKeys.TIME.value, 'size': 1},
            {'key': InputKeys.INPUT_STEER.value, 'size': 1},
            {'key': InputKeys.INPUT_GAS.value, 'size': 1},
            {'key': InputKeys.INPUT_BRAKE.value, 'size': 1},
            {'key': InputKeys.POSITION.value, 'size': 3},
            {'key': InputKeys.VELOCITY.value, 'size': 3},
        ],
        'categorical': [
            {'key': InputKeys.EVENT_TYPE.value, 'size': len(EventType)},
        ]
    })
    output_features: Dict[str, Any] = field(default_factory=lambda: {
        OutputKeys.TIME.value: {'size': 1},
        OutputKeys.INPUTS.value: {'size': 3},
        OutputKeys.POSITION.value: {'size': 3},
        OutputKeys.VELOCITY.value: {'size': 3},
        OutputKeys.EVENT_TYPE.value: {'size': len(EventType)},
        OutputKeys.BLOCK_NAME.value: {'size': None},  # Size to be set based on data
        OutputKeys.PAGE_NAME.value: {'size': None},  # Size to be set based on data
        OutputKeys.BLOCK_POSITION.value: {'size': 3},
        OutputKeys.BLOCK_DIRECTION.value: {'size': 5},
    })
    block_features: Dict[str, Any] = field(default_factory=lambda: {
        'PageName': {'size': None},  # Size to be set based on data
        'MaterialName': {'size': None},  # Size to be set based on data
        'Name': {'size': None},  # Size to be set based on data
        'Position': {'size': 3},
        'Direction': {'size': 5},
    })
    block_embedding_dim: int = 16
    hidden_sizes: Sequence[int] = (64, 32)

# Update EventType to include total number dynamically
EventTypeSize = len(EventType)

# Helper functions
def get_default_block_values() -> Dict[str, Any]:
    return {
        'PageName': -1,
        'MaterialName': -1,
        'Name': -1,
        'Position': np.zeros(3, dtype=np.float32),
        'Direction': -1
    }

def extract_numerical_fields(data: Dict[str, np.ndarray], config: ModelConfig) -> np.ndarray:
    numerical_fields = []
    for feature in config.input_features['numerical']:
        key = feature['key']
        size = feature['size']
        array = data[key]
        # Expand dimensions for scalar features
        if array.ndim == 2 and size == 1:
            array = array[..., None]  # Shape becomes (num_sequences, sequence_length, 1)
        elif array.ndim == 2 and size > 1:
            # Reshape array to (num_sequences, sequence_length, size)
            array = array.reshape(array.shape[0], array.shape[1], size)
        numerical_fields.append(array)
    return np.concatenate(numerical_fields, axis=-1)

def get_block_data_for_hashes(block_hashes: np.ndarray, tokenized_blocks: Dict[str, Dict[str, Any]], config: ModelConfig) -> Dict[str, np.ndarray]:
    default_values = get_default_block_values()
    num_blocks = len(block_hashes)
    field_values = {field: [] for field in default_values.keys()}

    for block_hash in block_hashes:
        block_hash_str = block_hash.decode("utf-8")
        block = tokenized_blocks.get(block_hash_str, None)
        for field in default_values.keys():
            value = block[field] if block is not None else default_values[field]
            field_values[field].append(value)
    for field in field_values:
        field_values[field] = np.array(field_values[field])
    return field_values

class DataProcessor:
    def __init__(self, manager: TrackmaniaDataManager, map_uid: str, config: ModelConfig):
        self.manager = manager
        self.map_uid = map_uid
        self.config = config
        self.tokenized_blocks = None
        self.global_stats = None
        self.tokenizers = None

    def prepare_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple]:
        raw_data = self.manager.prepare_data_for_training(self.map_uid, sequence_length=10, stride=1, test_split=0.2)
        all_block_hashes = self.extract_all_block_hashes(raw_data)
        self.tokenizers, self.tokenized_blocks = self.manager.get_tokenizers(self.map_uid, all_block_hashes)

        # Update config sizes based on data
        self.update_config_sizes()

        self.global_stats = self.get_or_compute_global_stats(raw_data['train_inputs'])
        train_data = self.process_and_normalize_data(raw_data['train_inputs'], raw_data['train_targets'])
        test_data = self.process_and_normalize_data(raw_data['test_inputs'], raw_data['test_targets'])
        return train_data, test_data, self.global_stats

    def update_config_sizes(self):
        # Update sizes in config based on tokenizers
        self.config.output_features[OutputKeys.BLOCK_NAME.value]['size'] = len(self.tokenizers.get('Name', {}))
        self.config.output_features[OutputKeys.PAGE_NAME.value]['size'] = len(self.tokenizers.get('PageName', {}))
        for block_feature in ['PageName', 'MaterialName', 'Name']:
            tokenizer_size = len(self.tokenizers.get(block_feature, {}))
            if tokenizer_size == 0:
                raise ValueError(f"Tokenizer for {block_feature} is empty.")
            self.config.block_features[block_feature]['size'] = tokenizer_size

    def extract_all_block_hashes(self, raw_data: Dict[str, Any]) -> set:
        block_hashes = set()
        for dataset in ['train_inputs', 'test_inputs']:
            for sequence in raw_data[dataset]:
                for event in sequence:
                    if event['EventType'] in (EventType.BLOCK_ENTER, EventType.BLOCK_EXIT):
                        block_hash = event['BlockHash'].decode('utf-8')
                        block_hashes.add(block_hash)
        return block_hashes

    def get_or_compute_global_stats(self, train_inputs: np.ndarray) -> Tuple[np.ndarray, ...]:
        if 'global_stats' in self.manager.file[f'maps/{self.map_uid}']:
            logger.info(f"Loading global stats from HDF5 for {self.map_uid}")
            return self.manager.load_global_stats(self.map_uid)
        else:
            train_numeric_data = extract_numerical_fields(train_inputs, self.config)
            global_stats = self.calculate_global_stats(train_numeric_data)
            self.manager.save_global_stats(self.map_uid, global_stats)
            return global_stats

    def calculate_global_stats(self, numeric_data: np.ndarray) -> Tuple[np.ndarray, ...]:
        position_indices = self.get_feature_indices('Position')
        velocity_indices = self.get_feature_indices('Velocity')

        position_data = numeric_data[..., position_indices].reshape(-1, 3)
        velocity_data = numeric_data[..., velocity_indices].reshape(-1, 3)

        global_position_mean = np.mean(position_data, axis=0)
        global_position_std = np.std(position_data, axis=0)
        global_velocity_mean = np.mean(velocity_data, axis=0)
        global_velocity_std = np.std(velocity_data, axis=0)
        return global_position_mean, global_position_std, global_velocity_mean, global_velocity_std

    def get_feature_indices(self, feature_key: str) -> slice:
        start_idx = 0
        for feature in self.config.input_features['numerical']:
            key = feature['key']
            size = feature['size']
            if key == feature_key:
                return slice(start_idx, start_idx + size)
            start_idx += size
        for feature in self.config.input_features['categorical']:
            key = feature['key']
            size = feature['size']
            if key == feature_key:
                return slice(start_idx, start_idx + size)
            start_idx += size
        raise ValueError(f"Feature {feature_key} not found in input features.")

    def process_and_normalize_data(self, inputs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        processed_inputs = self.preprocess_data(inputs)
        processed_targets = self.preprocess_data(targets)
        normalized_inputs = self.normalize_data(processed_inputs['inputs'])
        normalized_targets = self.normalize_data(processed_targets['inputs'])
        data = {
            'inputs': normalized_inputs,
            'event_type': processed_inputs['event_type'],
            'blocks': processed_inputs['blocks'],
            'targets': {
                'numerical': normalized_targets,
                'event_type': processed_targets['event_type'],
                'blocks': processed_targets['blocks']
            }
        }
        return data

    def preprocess_data(self, data: np.ndarray) -> Dict[str, Any]:
        event_data = extract_numerical_fields(data, self.config)
        event_types = data[InputKeys.EVENT_TYPE.value].astype(np.int32)
        
        num_event_types = self.config.output_features[OutputKeys.EVENT_TYPE.value]['size']
        event_types_one_hot = np.eye(num_event_types)[event_types]
        
        block_event_mask = np.isin(event_types, [EventType.BLOCK_ENTER, EventType.BLOCK_EXIT])
        block_indices = np.where(block_event_mask)
        block_hashes = data[InputKeys.BLOCK_HASH.value][block_indices]
        block_field_values = get_block_data_for_hashes(block_hashes, self.tokenized_blocks, self.config)

        num_sequences, sequence_length = event_types.shape
        block_data_shape = (num_sequences, sequence_length)
        block_data = {}
        for field, properties in self.config.block_features.items():
            size = properties['size']
            if field == 'Direction':
                block_data[field] = np.zeros(block_data_shape + (5,), dtype=np.float32)  # One-hot encoding
            elif size == 3:
                block_data[field] = np.zeros(block_data_shape + (3,), dtype=np.float32)
            else:
                block_data[field] = np.full(block_data_shape, get_default_block_values()[field], dtype=np.int32)

        for field in block_data.keys():
            if field == 'Direction':
                one_hot_direction = np.eye(5)[block_field_values[field]]
                block_data[field][block_indices[0], block_indices[1]] = one_hot_direction
            elif field == 'Position':
                block_data[field][block_indices[0], block_indices[1]] = block_field_values[field]
            else:
                block_data[field][block_indices[0], block_indices[1]] = block_field_values[field]
            block_data[field] = jnp.array(block_data[field])
        
        return {
            'inputs': jnp.array(event_data, dtype=jnp.float32), 
            'blocks': block_data,
            'event_type': jnp.array(event_types_one_hot)
        }

    def normalize_data(self, data: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        global_position_mean, global_position_std, global_velocity_mean, global_velocity_std = self.global_stats
        time_idx = self.get_feature_indices(InputKeys.TIME.value)
        position_idx = self.get_feature_indices('Position')
        velocity_idx = self.get_feature_indices('Velocity')

        time = data[..., time_idx]
        numerical_features = data[..., :]

        time_normalized = time - jnp.min(time, axis=-2, keepdims=True)
        position_normalized = (data[..., position_idx] - global_position_mean) / (global_position_std + 1e-8)
        velocity_normalized = (data[..., velocity_idx] - global_velocity_mean) / (global_velocity_std + 1e-8)

        # Reconstruct normalized numerical data
        normalized_numerical = jnp.concatenate([
            time_normalized,
            numerical_features[..., 1:position_idx.start],
            position_normalized,
            velocity_normalized
        ], axis=-1)

        return normalized_numerical

class BasicTrackmaniaNN(nn.Module):
    config: ModelConfig

    def setup(self):
        block_embeddings = {}
        for feature_name, properties in self.config.block_features.items():
            if properties['size'] and properties['size'] > 0 and feature_name not in ['Position', 'Direction']:
                block_embeddings[feature_name] = nn.Embed(
                    num_embeddings=properties['size'],
                    features=self.config.block_embedding_dim
                )
        self.dense_layers = [nn.Dense(size) for size in self.config.hidden_sizes]
        self.block_embeddings = block_embeddings

    @nn.compact
    def __call__(self, x, block_data, event_type, train: bool = True):

        # Embeddings
        block_embeds = []
        for feature_name, embedding in self.block_embeddings.items():
            block_feature = block_data[feature_name]
            embedded = embedding(block_feature)
            block_embeds.append(embedded)
        
        # Add Position and Direction (already in the correct format)
        block_embeds.append(block_data['Position'])
        block_embeds.append(block_data['Direction'])
        
        block_embeddings_concat = jnp.concatenate(block_embeds, axis=-1)

        x = jnp.concatenate([x, block_embeddings_concat, event_type], axis=-1)

        # Hidden Layers
        for i, layer in enumerate(self.dense_layers):
            x = nn.relu(layer(x))

        # Outputs
        outputs = {}
        for output_name, properties in self.config.output_features.items():
            size = properties['size']
            outputs[output_name] = nn.Dense(size)(x)
        return outputs

def create_train_state(rng, model, learning_rate, input_shape, block_shapes):
    batch_size, seq_length, _ = input_shape
    dummy_input = jnp.ones(input_shape)
    num_event_types = model.config.output_features[OutputKeys.EVENT_TYPE.value]['size']
    dummy_event_type = jnp.ones((batch_size, seq_length, num_event_types), dtype=jnp.float32)
    dummy_block_data = {}
    for key, properties in model.config.block_features.items():
        size = properties['size']
        if key == 'Direction':
            dummy_block_data[key] = jnp.ones((batch_size, seq_length, 5), dtype=jnp.float32)
        elif size == 3:
            dummy_block_data[key] = jnp.ones((batch_size, seq_length, 3), dtype=jnp.float32)
        else:
            dummy_block_data[key] = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    params = model.init(rng, dummy_input, dummy_block_data, dummy_event_type)['params']
    tx = optax.chain(optax.adam(learning_rate), optax.clip_by_global_norm(1.0))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Function to create batches
def create_batches(data: Dict[str, Any], batch_size: int):
    num_samples = data['inputs'].shape[0]
    for i in range(0, num_samples, batch_size):
        batch = {
            'inputs': data['inputs'][i:i+batch_size],
            'event_type': data['event_type'][i:i+batch_size],
            'blocks': {k: v[i:i+batch_size] for k, v in data['blocks'].items()},
            'targets': {
                'numerical': data['targets']['numerical'][i:i+batch_size],
                'event_type': data['targets']['event_type'][i:i+batch_size],
                'blocks': {k: v[i:i+batch_size] for k, v in data['targets']['blocks'].items()}
            }
        }
        yield batch


@jax.jit
def custom_loss(predictions, targets, loss_weights, output_features: Dict[str, Any]):
    total_loss = 0.0
    for output_name, properties in output_features.items():
        size = properties['size']
        pred = predictions[output_name]
        if output_name in targets['numerical']:
            true = targets['numerical'][..., :size]
        elif output_name in targets:
            true = targets[output_name]
        else:
            continue
        
        if output_name in [OutputKeys.BLOCK_NAME.value, OutputKeys.PAGE_NAME.value]:
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, true))
        elif output_name in [OutputKeys.EVENT_TYPE.value, OutputKeys.BLOCK_DIRECTION.value]:
            loss = jnp.mean(optax.softmax_cross_entropy(pred, true))
        else:
            # Mean squared error loss
            loss = jnp.mean((pred - true) ** 2)
        
        total_loss += loss_weights.get(output_name, 1.0) * loss
    return total_loss

@jax.jit
def train_step(state, batch, loss_weights, output_features: Dict[str, Any]):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['inputs'], batch['blocks'], batch['event_type'])
        return custom_loss(predictions, batch['targets'], loss_weights, output_features)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch, loss_weights, output_features: Dict[str, Any]):
    predictions = state.apply_fn({'params': state.params}, batch['inputs'], batch['blocks'], batch['event_type'])
    return custom_loss(predictions, batch['targets'], loss_weights, output_features)


def calculate_accuracy(predictions, targets, config):
    accuracies = {}
    
    # Event Type Accuracy
    event_type_pred = jnp.argmax(predictions[OutputKeys.EVENT_TYPE.value], axis=-1)
    event_type_true = jnp.argmax(targets['event_type'], axis=-1)
    event_type_acc = jnp.mean(event_type_pred == event_type_true)
    accuracies['event_type'] = float(event_type_acc)
    
    # Block Name Accuracy
    block_name_pred = jnp.argmax(predictions[OutputKeys.BLOCK_NAME.value], axis=-1)
    block_name_true = targets['blocks']['Name']
    block_name_acc = jnp.mean(block_name_pred == block_name_true)
    accuracies['block_name'] = float(block_name_acc)

    # Page Name Accuracy
    page_name_pred = jnp.argmax(predictions[OutputKeys.PAGE_NAME.value], axis=-1)
    page_name_true = targets['blocks']['PageName']
    page_name_acc = jnp.mean(page_name_pred == page_name_true)
    accuracies['page_name'] = float(page_name_acc)
    
    # Block Position Accuracy (using a threshold)
    position_threshold = 0.1  # Adjust as needed
    block_position_pred = predictions[OutputKeys.BLOCK_POSITION.value]
    block_position_true = targets['blocks']['Position']
    position_correct = jnp.sum(jnp.abs(block_position_pred - block_position_true) < position_threshold, axis=-1) == 3
    block_position_acc = jnp.mean(position_correct)
    accuracies['block_position'] = float(block_position_acc)
    
    # Block Direction Accuracy
    block_direction_pred = jnp.argmax(predictions[OutputKeys.BLOCK_DIRECTION.value], axis=-1)
    block_direction_true = jnp.argmax(targets['blocks']['Direction'], axis=-1)
    block_direction_acc = jnp.mean(block_direction_pred == block_direction_true)
    accuracies['block_direction'] = float(block_direction_acc)
    
    return accuracies

def evaluate_accuracy(state, data, config, batch_size=32):
    all_predictions = []
    all_targets = []
    
    for batch in create_batches(data, batch_size):
        predictions = state.apply_fn({'params': state.params}, batch['inputs'], batch['blocks'], batch['event_type'])
        all_predictions.append(predictions)
        all_targets.append(batch['targets'])
    
    # Combine all batches
    combined_predictions = {k: jnp.concatenate([p[k] for p in all_predictions], axis=0) for k in all_predictions[0]}
    combined_targets = {
        'event_type': jnp.concatenate([t['event_type'] for t in all_targets], axis=0),
        'blocks': {
            'Name': jnp.concatenate([t['blocks']['Name'] for t in all_targets], axis=0),
            'PageName': jnp.concatenate([t['blocks']['PageName'] for t in all_targets], axis=0),
            'Position': jnp.concatenate([t['blocks']['Position'] for t in all_targets], axis=0),
            'Direction': jnp.concatenate([t['blocks']['Direction'] for t in all_targets], axis=0),
        }
    }
    
    return calculate_accuracy(combined_predictions, combined_targets, config)




config = ModelConfig()

manager = TrackmaniaDataManager('trackmania_dataset.h5')
map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'
data_processor = DataProcessor(manager, map_uid, config)
train_data, test_data, global_stats = data_processor.prepare_data()

# Initialize Model
model = BasicTrackmaniaNN(config=config)

# Create training state
rng = jax.random.PRNGKey(0)
input_shape = train_data['inputs'].shape
block_shapes = {key: value.shape[1:] for key, value in train_data['blocks'].items()}
state = create_train_state(rng, model, learning_rate=0.01, input_shape=input_shape, block_shapes=block_shapes)

# Define loss weights
loss_weights = {
    OutputKeys.POSITION.value: 0.01,
    OutputKeys.VELOCITY.value: 0.01,
    OutputKeys.EVENT_TYPE.value: 1,
    OutputKeys.TIME.value: 0.00001,
    OutputKeys.BLOCK_NAME.value: 1.0,
    OutputKeys.PAGE_NAME.value: 1.0,
    OutputKeys.BLOCK_POSITION.value: 0.01,
    OutputKeys.BLOCK_DIRECTION.value: 1,
    OutputKeys.INPUTS.value: 1.0
}

# Training loop
num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
    batch_losses = []
    for batch in create_batches(train_data, batch_size):
        state, loss = train_step(state, batch, loss_weights, config.output_features)
        batch_losses.append(loss)
    train_loss = jnp.mean(jnp.array(batch_losses))

    # Evaluation (optional)
    eval_losses = []
    for batch in create_batches(test_data, batch_size):
        loss = eval_step(state, batch, loss_weights, config.output_features)
        eval_losses.append(loss)
    test_loss = jnp.mean(jnp.array(eval_losses))

    
    test_accuracy = evaluate_accuracy(state, test_data, config)

    logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    logger.info(f"  Accuracies: 'Event Type': {test_accuracy['event_type']:.4f}, 'Block Name': {test_accuracy['block_name']:.4f}, 'Page Name': {test_accuracy['page_name']:.4f}, 'Block Position': {test_accuracy['block_position']:.4f}, 'Block Direction': {test_accuracy['block_direction']:.4f}")

# Optional: Save the trained model parameters
# save_path = 'trained_model.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(state.params, f)