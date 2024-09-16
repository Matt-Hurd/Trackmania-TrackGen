import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import List, Sequence, Tuple, Dict, Any
import numpy as np
import logging
from dataclasses import dataclass, field
from collections import OrderedDict
from enums import EncodingType, EventType
from features import FeatureInfo, Features

from data_manager import TrackmaniaDataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    block_embedding_dim: int = 16
    hidden_sizes: Sequence[int] = (64, 32)


def get_default_block_values() -> Dict[str, Any]:
    return {
        Features.BLOCK_PAGE_NAME.name: 0,
        Features.BLOCK_MATERIAL_NAME.name: 0,
        Features.BLOCK_NAME.name: 0,
        Features.BLOCK_POSITION.name: np.zeros(3, dtype=np.float32),
        Features.BLOCK_DIRECTION.name: 5
    }

def extract_numerical_fields(data: Dict[str, np.ndarray]) -> np.ndarray:
    numerical_fields = []
    for feature in Features.get_numerical_features():
        key = feature.name
        size = feature.size
        array = data[key]
        if array.ndim == 2 and size == 1:
            array = array[..., None]
        elif array.ndim == 2 and size > 1:
            array = array.reshape(array.shape[0], array.shape[1], size)
        numerical_fields.append(array)
    return np.concatenate(numerical_fields, axis=-1)

def get_block_data_for_hashes(block_hashes: np.ndarray, tokenized_blocks: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    default_values = get_default_block_values()
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
    def __init__(self, manager: TrackmaniaDataManager, map_uid: str):
        self.manager = manager
        self.map_uid = map_uid
        self.config = config
        self.tokenized_blocks = None
        self.global_stats = None
        self.tokenizers = None

    def prepare_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        raw_data = self.manager.prepare_data_for_training(self.map_uid, sequence_length=10, stride=1, test_split=0.2)
        all_block_hashes = self.extract_all_block_hashes(raw_data)
        self.tokenizers, self.tokenized_blocks = self.manager.get_tokenizers(self.map_uid, all_block_hashes)

        # Update config sizes based on data
        self.update_config_sizes()

        self.global_stats = self.get_or_compute_global_stats(raw_data['train_inputs'])

        train = {
            "inputs": self.preprocess_data(raw_data['train_inputs']),
            "targets": self.preprocess_data(raw_data['train_targets']),
        }
        test = {
            "inputs": self.preprocess_data(raw_data['test_inputs']),
            "targets": self.preprocess_data(raw_data['test_targets']),
        }
        return train, test, self.global_stats

    def update_config_sizes(self):
        for block_feature in Features.get_block_features():
            if block_feature.encoding != EncodingType.TOKENIZED:
                continue
            Features.set_feature_size(block_feature, len(self.tokenizers.get(block_feature.name, {})))
            if block_feature.size == 0:
                raise ValueError(f"Tokenizer for {block_feature} is empty.")

    def extract_all_block_hashes(self, raw_data: Dict[str, Any]) -> set:
        block_hashes = set()
        for dataset in ['train_inputs', 'test_inputs']:
            for sequence in raw_data[dataset]:
                for event in sequence:
                    if event['EventType'] in (EventType.BLOCK_ENTER, EventType.BLOCK_EXIT):
                        block_hash = event['BlockHash'].decode('utf-8')
                        block_hashes.add(block_hash)
        return block_hashes

    def get_or_compute_global_stats(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.global_stats is None:
            global_position_mean = np.mean(data[Features.POSITION.name].astype(np.float32), axis=(0, 1))
            global_position_std = np.std(data[Features.POSITION.name].astype(np.float32), axis=(0, 1))
            global_velocity_mean = np.mean(data[Features.VELOCITY.name].astype(np.float32), axis=(0, 1))
            global_velocity_std = np.std(data[Features.VELOCITY.name].astype(np.float32), axis=(0, 1))
            
            self.global_stats = (global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)
        
        return self.global_stats

    def preprocess_data(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        self.normalize_data(data)

        # Collect all non-block features
        features = []
        for feature in Features.get_all_features():
            if feature.is_block_feature:
                continue

            if feature.encoding == EncodingType.ONE_HOT:
                # Convert to one-hot vectors
                feature_data = data[feature.name].astype(np.int32)
                one_hot = np.eye(feature.size)[feature_data]  # Shape: (batch, timesteps, size)
                features.append(one_hot)
            else:
                # Continuous or other encoded features
                feature_data = data[feature.name].astype(np.float32)
                if feature_data.ndim != 1:
                    feature_data = feature_data.reshape(feature_data.shape[0], feature_data.shape[1], -1)
                features.append(feature_data)

        # Concatenate all non-block features along the last axis (features)
        x = np.concatenate(features, axis=-1)  # Shape: (batch, timesteps, total_features)

        # Handle block features as before
        event_types = data[Features.EVENT_TYPE.name].astype(np.int32)
        block_event_mask = np.isin(event_types, [EventType.BLOCK_ENTER, EventType.BLOCK_EXIT])
        block_indices = np.where(block_event_mask)
        block_hashes = data[Features.BLOCK_HASH.name][block_indices]
        block_field_values = get_block_data_for_hashes(block_hashes, self.tokenized_blocks)

        # Initialize block_data with default values
        block_data = {}
        num_samples, num_timesteps = x.shape[:2]
        for feature in Features.get_block_features():
            if feature.encoding == EncodingType.NONE:
                continue
            if feature.encoding == EncodingType.ONE_HOT:
                feature_dim = feature.size
                block_data[feature.name] = np.full(
                    (num_samples, num_timesteps, feature_dim),
                    get_default_block_values()[feature.name],
                    dtype=np.float32
                )
            elif feature.encoding == EncodingType.TOKENIZED:
                block_data[feature.name] = np.full(
                    (num_samples, num_timesteps),
                    get_default_block_values()[feature.name],  # Should be integer indices
                    dtype=np.int32
                )
            else:
                block_data[feature.name] = np.full(
                    (num_samples, num_timesteps, feature.size),
                    get_default_block_values()[feature.name],
                    dtype=np.float32
                )

        # Assign block data to the corresponding timesteps
        for idx in range(len(block_hashes)):
            sample_idx, timestep_idx = block_indices[0][idx], block_indices[1][idx]
            for feature in Features.get_block_features():
                if feature.encoding == EncodingType.NONE:
                    continue
                value = block_field_values[feature.name][idx]
                if feature.encoding == EncodingType.ONE_HOT:
                    block_data[feature.name][sample_idx, timestep_idx] = np.eye(feature.size)[value]
                else:
                    block_data[feature.name][sample_idx, timestep_idx] = value

        # Debugging: Log shapes of data and blocks
        logger.info(f"Preprocessed data shape: {x.shape}")
        for key, value in block_data.items():
            logger.info(f"Block feature '{key}' shape: {value.shape}")

        return {
            "data": x,
            "blocks": block_data
        }

    def normalize_data(self, data: np.ndarray) -> None:
        global_position_mean, global_position_std, global_velocity_mean, global_velocity_std = self.global_stats
        
        # Normalize time
        time = data[Features.TIME.name]
        data[Features.TIME.name] = time - np.min(time, axis=1, keepdims=True)

        # Normalize position
        position = data[Features.POSITION.name]
        data[Features.POSITION.name] = (position - global_position_mean) / (global_position_std + 1e-8)

        # Normalize velocity
        velocity = data[Features.VELOCITY.name]
        data[Features.VELOCITY.name] = (velocity - global_velocity_mean) / (global_velocity_std + 1e-8)

class BasicTrackmaniaNN(nn.Module):
    config: ModelConfig

    def setup(self):
        block_embeddings = {}
        for feature in Features.get_block_features():
            if feature.encoding == EncodingType.TOKENIZED:
                block_embeddings[feature.name] = nn.Embed(
                    num_embeddings=feature.size,  # Vocabulary size
                    features=self.config.block_embedding_dim
                )
        self.block_embeddings = block_embeddings
        self.dense_layers = [nn.Dense(size) for size in self.config.hidden_sizes]

    @nn.compact
    def __call__(self, x, block_data, train: bool = True):
        # Embeddings
        block_embeds = []
        
        for feature in Features.get_block_features():
            if feature.encoding == EncodingType.NONE:
                continue
            if feature.encoding == EncodingType.TOKENIZED:
                embedding = self.block_embeddings[feature.name]
                block_feature = block_data[feature.name]  # Shape: (batch, timesteps)
                embedded = embedding(block_feature)  # Shape: (batch, timesteps, embedding_dim)
                block_embeds.append(embedded)
            elif feature.encoding == EncodingType.ONE_HOT:
                block_feature = block_data[feature.name]  # Shape: (batch, timesteps, feature_dim)
                block_embeds.append(block_feature)
            else:
                # Handle other encodings if necessary
                block_feature = block_data[feature.name]
                block_embeds.append(block_feature)
        
        # Concatenate along the feature dimension (last axis)
        if block_embeds:
            block_embeddings_concat = jnp.concatenate(block_embeds, axis=-1)  # Shape: (batch, timesteps, total_block_features)
            x = jnp.concatenate([x, block_embeddings_concat], axis=-1)  # Shape: (batch, timesteps, x_features + block_features)
        else:
            x = x  # No block embeddings to concatenate

        # Hidden Layers
        for layer in self.dense_layers:
            x = nn.relu(layer(x))

        # Outputs
        outputs = OrderedDict()
        for feature in Features.get_all_features():
            if feature.encoding == EncodingType.NONE:
                continue
            outputs[feature.name] = nn.Dense(feature.size)(x)
        return outputs
    
def create_train_state(rng, model, learning_rate, input_shape, block_shapes):
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    dummy_block_data = {}
    
    # Ensure block_shapes include all necessary dimensions (batch_size, timesteps, num_tokens or feature_dim)
    for block_feature in Features.get_block_features():
        if block_feature.encoding == EncodingType.NONE:
            continue
        if block_feature.encoding == EncodingType.TOKENIZED:
            dummy_block_data[block_feature.name] = jnp.ones(
                block_shapes[block_feature.name], dtype=jnp.int32
            )
        elif block_feature.encoding == EncodingType.ONE_HOT:
            dummy_block_data[block_feature.name] = jnp.eye(block_feature.size)[
                jnp.ones((block_shapes[block_feature.name][0], block_shapes[block_feature.name][1]), dtype=jnp.int32)
            ].astype(jnp.float32)
        else:
            dummy_block_data[block_feature.name] = jnp.ones(
                block_shapes[block_feature.name], dtype=jnp.float32
            )
    
    # Debugging: Log shapes of dummy inputs
    logger.debug(f"Dummy input shape: {dummy_input.shape}")
    for key, value in dummy_block_data.items():
        logger.debug(f"Dummy block '{key}' shape: {value.shape}")
    
    params = model.init(rng, dummy_input, dummy_block_data)['params']
    tx = optax.chain(optax.adam(learning_rate), optax.clip_by_global_norm(1.0))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Function to create batches
def create_batches(data: Dict[str, Any], batch_size: int):
    num_samples = data['inputs']['data'].shape[0]
    for i in range(0, num_samples, batch_size):
        batch = {
            'inputs': {
                'data': data['inputs']['data'][i:i+batch_size],
                'blocks': {key: value[i:i+batch_size] for key, value in data['inputs']['blocks'].items()}
            },
            'targets': {
                'data': data['targets']['data'][i:i+batch_size],
                'blocks': {key: value[i:i+batch_size] for key, value in data['targets']['blocks'].items()}
            }
        }
        yield batch


@jax.jit
def custom_loss(predictions, targets, loss_weights):
    total_loss = 0.0
    for feature in Features.get_all_features():
        if feature.name not in loss_weights or feature.encoding == EncodingType.NONE:
            continue

        index = Features.get_feature_index(feature, False)
        pred = predictions[feature.name]

        if feature.is_block_feature:
            true = targets['blocks'][feature.name]
        else:
            true = targets['data'][..., index]

        if feature.encoding == EncodingType.ONE_HOT:
            loss = jnp.mean(optax.softmax_cross_entropy(pred, true))
        elif feature.encoding == EncodingType.TOKENIZED:
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, true))
        else:
            loss = jnp.mean((pred - true) ** 2)

        total_loss += loss_weights[feature.name] * loss

    return total_loss

@jax.jit
def train_step(state, batch, loss_weights):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['inputs']['data'], batch['inputs']['blocks'])
        return custom_loss(predictions, batch['targets'], loss_weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch, loss_weights: Dict[str, Any]):
    predictions = state.apply_fn({'params': state.params}, batch['inputs']['data'], batch['inputs']['blocks'])
    return custom_loss(predictions, batch['targets'], loss_weights)

@jax.jit
def calculate_accuracy(predictions: Dict[str, jnp.ndarray], targets: Dict[str, Any]):
    accuracies = {}
    
    # Retrieve feature slices for non-block features
    feature_slices = Features.get_feature_slices()
    
    for feature in Features.get_all_features():
        if feature.name not in loss_weights or feature.encoding == EncodingType.NONE:
            continue

        pred = predictions[feature.name]

        if feature.is_block_feature:
            true = targets['blocks'][feature.name]
        else:
            # Extract the slice corresponding to the current feature
            feature_slice = feature_slices.get(feature.name)
            if feature_slice is None:
                raise ValueError(f"No slice found for feature {feature.name}")
            true = targets['data'][feature.name]

        # Debugging: Print feature, prediction shape, and true shape

        if feature.encoding == EncodingType.ONE_HOT:
            # For ONE_HOT features, compute accuracy by comparing argmax
            pred_labels = jnp.argmax(pred, axis=-1)
            true_labels = jnp.argmax(true, axis=-1)
            acc = jnp.mean(pred_labels == true_labels)
            accuracies[feature.name] = acc
        
        elif feature.encoding == EncodingType.TOKENIZED:
            # For TOKENIZED features, true should be integer indices
            pred_labels = jnp.argmax(pred, axis=-1)
            true_labels = true  # Should already be integer indices
            acc = jnp.mean(pred_labels == true_labels)
            accuracies[feature.name] = acc
        
        elif feature.encoding == EncodingType.NUMERICAL:
            # For NUMERICAL features, use a threshold-based accuracy
            position_threshold = 1.0  # Adjust as needed
            # Assuming numerical features have multiple dimensions (e.g., x, y, z)
            correct = jnp.all(jnp.abs(pred - true) < position_threshold, axis=-1)
            acc = jnp.mean(correct)
            accuracies[feature.name] = acc
        
        else:
            # Handle other encodings if necessary
            pass
    
    return accuracies

@jax.jit
def evaluate_accuracy(state, data, batch_size=32):
    all_predictions = []
    all_targets = []
    
    for batch in create_batches(data, batch_size):
        # Forward pass to get predictions
        predictions = state.apply_fn({'params': state.params}, batch['inputs']['data'], batch['inputs']['blocks'])
        all_predictions.append(predictions)
        all_targets.append(batch['targets'])
    
    # Initialize combined_predictions keys
    combined_predictions = {key: [] for key in all_predictions[0]}
    combined_targets = {
        'data': {feature.name: [] for feature in Features.get_all_features() if not feature.is_block_feature},
        'blocks': {feature.name: [] for feature in Features.get_all_features() if feature.is_block_feature}
    }
    
    # Retrieve feature slices for non-block features
    feature_slices = Features.get_feature_slices()
    
    # Aggregate predictions and targets
    for preds, targs in zip(all_predictions, all_targets):
        for key in preds:
            combined_predictions[key].append(preds[key])
        for feature in Features.get_all_features():
            if feature.encoding == EncodingType.NONE:
                continue
            if feature.is_block_feature:
                combined_targets['blocks'][feature.name].append(targs['blocks'][feature.name])
            else:
                feature_slice = feature_slices.get(feature.name)
                if feature_slice is None:
                    raise ValueError(f"No slice found for feature {feature.name}")
                # Extract the specific feature data from targs['data'] via slicing
                true_feature = targs['data'][..., feature_slice]
                combined_targets['data'][feature.name].append(true_feature)
    
    # Concatenate predictions
    for key in combined_predictions:
        combined_predictions[key] = jnp.concatenate(combined_predictions[key], axis=0)
    
    # Concatenate targets
    for feature in Features.get_all_features():
        if feature.encoding == EncodingType.NONE:
            continue
        if feature.is_block_feature:
            combined_targets['blocks'][feature.name] = jnp.concatenate(combined_targets['blocks'][feature.name], axis=0)
        else:
            combined_targets['data'][feature.name] = jnp.concatenate(combined_targets['data'][feature.name], axis=0)
    
    # Compute accuracies
    accuracies = calculate_accuracy(combined_predictions, combined_targets)
    
    return accuracies

config = ModelConfig()

manager = TrackmaniaDataManager('trackmania_dataset.h5')
map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'
data_processor = DataProcessor(manager, map_uid)
train_data, test_data, global_stats = data_processor.prepare_data()

# Initialize Model
model = BasicTrackmaniaNN(config=config)

# Create training state
rng = jax.random.PRNGKey(0)
input_shape = train_data['inputs']['data'].shape
block_shapes = {key: value.shape for key, value in train_data['inputs']['blocks'].items()}
state = create_train_state(rng, model, learning_rate=0.01, input_shape=input_shape, block_shapes=block_shapes)

# Define loss weights
loss_weights = {
    Features.POSITION.name: 0.01,
    Features.VELOCITY.name: 0.01,
    Features.EVENT_TYPE.name: 10,
    Features.TIME.name: 0.0001,
    Features.BLOCK_NAME.name: 5.0,
    Features.BLOCK_PAGE_NAME.name: 5.0,
    Features.BLOCK_POSITION.name: 0.01,
    Features.BLOCK_DIRECTION.name: 1,
    Features.INPUT_STEER.name: 0.01,
    Features.INPUT_GAS_PEDAL.name: 0.1,
    Features.INPUT_BRAKE_PEDAL.name: 0.1
}

# Training loop
num_epochs = 1000
batch_size = 64

for epoch in range(num_epochs):
    batch_losses = []
    for batch in create_batches(train_data, batch_size):
        state, loss = train_step(state, batch, loss_weights)
        batch_losses.append(loss)
    train_loss = jnp.mean(jnp.array(batch_losses))
    logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    # Evaluation (optional)
    # eval_losses = []
    # for batch in create_batches(test_data, batch_size):
    #     loss = eval_step(state, batch, loss_weights, config.output_features)
    #     eval_losses.append(loss)
    # test_loss = jnp.mean(jnp.array(eval_losses))
    
    test_accuracy = evaluate_accuracy(state, test_data)

    # logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    out = f"  Accuracies: "
    for key, value in test_accuracy.items():
        out += f"'{key}': {value:.4f}, "
    logger.info(out)
