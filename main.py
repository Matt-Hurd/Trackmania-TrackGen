import logging
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import List, Sequence, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from enums import EncodingType, EventType
from features import Features
from flax.training import checkpoints

from config import loss_weights

from data_manager import TrackmaniaDataManager
from positional_encoding import PositionalEncoding
from predict import collect_and_save_predictions, predict_single_batch
from transformer_blocks import TransformerConfig, TransformerEncoderBlock

import os

CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)
CHECKPOINT_PREFIX = 'train_state'
CHECKPOINT_MAX_TO_KEEP = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    block_embedding_dim: int = 64
    hidden_sizes: Sequence[int] = (64, 32)
    d_model: int = 128  # Transformer model dimension
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of Transformer layers
    mlp_dim: int = 512  # Dimension of the MLP in Transformer
    dropout_rate: float = 0.1  # Dropout rate
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.bfloat16
    deterministic: bool = False
    warmup_epochs = 5
    num_epochs = 1000


def get_default_block_values() -> Dict[str, Any]:
    return {
        Features.BLOCK_PAGE_NAME.name: 0,
        Features.BLOCK_MATERIAL_NAME.name: 0,
        Features.BLOCK_NAME.name: 0,
        Features.BLOCK_POSITION.name: np.zeros(3, dtype=np.float32),
        Features.BLOCK_DIRECTION.name: 0
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
    def __init__(self, manager: TrackmaniaDataManager, map_uid: str, config: ModelConfig):
        self.manager = manager
        self.map_uid = map_uid
        self.config = config
        self.tokenized_blocks = None
        self.global_stats = None
        self.tokenizers = None

    def prepare_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        raw_data = self.manager.prepare_data_for_training(self.map_uid, sequence_length=32, stride=1, test_split=0.2)
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
                    if feature == Features.BLOCK_DIRECTION: # HACK
                        block_data[feature.name][sample_idx, timestep_idx] = np.eye(feature.size)[value + 1]
                    else:
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
        data[Features.POSITION.name] = (position - global_position_mean) # / (global_position_std + 1e-8)

        # Normalize velocity
        velocity = data[Features.VELOCITY.name]
        data[Features.VELOCITY.name] = (velocity - global_velocity_mean) # / (global_velocity_std + 1e-8)

class BasicTrackmaniaNN(nn.Module):
    config: ModelConfig
    max_seq_length: int = 32  # Adjust based on your data

    def setup(self):
        # Block embeddings
        self.block_embeddings = {
            feature.name: nn.Embed(
                num_embeddings=feature.size,
                features=self.config.block_embedding_dim
            )
            for feature in Features.get_block_features()
            if feature.encoding == EncodingType.TOKENIZED
        }
        
        # Calculate total block embedding dimension
        # total_block_emb_dim = sum(
        #     self.config.block_embedding_dim if feature.encoding == EncodingType.TOKENIZED else feature.size
        #     for feature in Features.get_block_features() if feature.encoding != EncodingType.NONE
        # )
        
        # Initial Dense layer to project input features to d_model
        self.input_projection = nn.Dense(self.config.d_model, use_bias=False)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=self.config.d_model)
        
        # Transformer Encoder Layers
        transformer_config = TransformerConfig(
            num_heads=self.config.num_heads,
            d_model=self.config.d_model,
            mlp_dim=self.config.mlp_dim,
            dropout_rate=self.config.dropout_rate,
            attention_dropout_rate=self.config.attention_dropout_rate,
            dtype=self.config.dtype,
            deterministic=self.config.deterministic
        )
        self.transformer_layers = [
            TransformerEncoderBlock(config=transformer_config) for _ in range(self.config.num_layers)
        ]
        
        # Final Dense Layers
        self.dense_layers = [nn.Dense(size) for size in self.config.hidden_sizes]
        
        # Define Output Dense Layers using ModuleDict
        output_features = [feature for feature in Features.get_all_features() if feature.encoding != EncodingType.NONE]
        self.output_layers = {
            feature.name: nn.Dense(feature.size)
            for feature in output_features
        }

    def __call__(self, x, block_data, train: bool = True):
        seq_length = x.shape[1]
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
        
        # Concatenate block embeddings along the feature dimension
        if block_embeds:
            block_embeddings_concat = jnp.concatenate(block_embeds, axis=-1)  # Shape: (batch, timesteps, total_block_features)
            x = jnp.concatenate([x, block_embeddings_concat], axis=-1)  # Shape: (batch, timesteps, x_features + block_features)
        else:
            x = x  # No block embeddings to concatenate
        
        # Project input to d_model
        x = self.input_projection(x)  # Shape: (batch, timesteps, d_model)
        
        # Apply Positional Encoding
        x = self.positional_encoding(x[:, :seq_length])  # Shape: (batch, timesteps, d_model)
        
        # Apply Transformer Encoder Layers
        for layer in self.transformer_layers:
            x = layer(x, train=train)  # Shape: (batch, timesteps, d_model)
        
        # Hidden Layers
        for layer in self.dense_layers:
            x = nn.relu(layer(x))  # Shape: (batch, timesteps, hidden_size)
        
        # Outputs
        outputs = OrderedDict()
        for feature_name in self.output_layers:
            outputs[feature_name] = self.output_layers[feature_name](x)  # Shape: (batch, timesteps, feature.size)
        return outputs

def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch)
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])
    return schedule_fn

def create_train_state(rngs, model, learning_rate, input_shape, block_shapes):
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
    
    params = model.init(rngs, dummy_input, dummy_block_data)['params']
    tx = optax.chain(optax.adam(learning_rate), optax.clip_by_global_norm(1.0))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Function to create batches
def create_batches(data: Dict[str, Any], batch_size: int):
    num_samples = data['inputs']['data'].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = {
            'inputs': {
                'data': data['inputs']['data'][batch_indices],
                'blocks': {key: value[batch_indices] for key, value in data['inputs']['blocks'].items()}
            },
            'targets': {
                'data': data['targets']['data'][batch_indices],
                'blocks': {key: value[batch_indices] for key, value in data['targets']['blocks'].items()}
            }
        }
        yield batch


@jax.jit
def custom_loss(predictions, targets, loss_weights):
    total_loss = 0.0
    event_types = targets['data'][..., Features.get_feature_index(Features.EVENT_TYPE, False)]  # Get event types
    block_event_types = jnp.array([EventType.BLOCK_ENTER, EventType.BLOCK_EXIT])

    for feature in Features.get_all_features():
        if feature.name not in loss_weights or feature.encoding == EncodingType.NONE:
            continue

        index = Features.get_feature_index(feature, False)
        pred = predictions[feature.name]

        if feature.is_block_feature:
            true = targets['blocks'][feature.name]
            block_event_mask = jnp.isin(event_types, block_event_types)
        else:
            true = targets['data'][..., index]
            block_event_mask = jnp.ones_like(true, dtype=jnp.float32)

        if feature.encoding == EncodingType.ONE_HOT:
            loss = jnp.mean(optax.softmax_cross_entropy(pred, true))
        elif feature.encoding == EncodingType.TOKENIZED:
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, true))
        else:
            loss = jnp.mean((pred - true) ** 2)

        masked_loss = jnp.mean(loss * block_event_mask)

        total_loss += loss_weights[feature.name] * masked_loss
        # print(f"Feature: {feature.name}, Loss: {jax.device_get(loss).item() * loss_weights[feature.name]}")

    return total_loss

@jax.jit
def train_step(state, batch, loss_weights, rng_key):
    def loss_fn(params):
        rngs = {'dropout': rng_key}
        predictions = state.apply_fn({'params': params}, batch['inputs']['data'], batch['inputs']['blocks'], rngs=rngs)
        return custom_loss(predictions, batch['targets'], loss_weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch, loss_weights: Dict[str, Any], rng_key):
    rngs = {'dropout': rng_key}
    predictions = state.apply_fn({'params': state.params}, batch['inputs']['data'], batch['inputs']['blocks'], rngs=rngs)
    return custom_loss(predictions, batch['targets'], loss_weights)

@jax.jit
def calculate_accuracy(predictions: Dict[str, jnp.ndarray], targets: Dict[str, Any]):
    accuracies = {}

    
    for feature in Features.get_all_features():
        if feature.name not in loss_weights or feature.encoding == EncodingType.NONE:
            continue

        pred = predictions[feature.name]

        if feature.is_block_feature:
            true = targets['blocks'][feature.name]
        else:
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
            # For NUMERICAL features, calculate average distance
            distances = jnp.abs(pred - true)
            avg_distance = jnp.mean(distances)
            accuracies[feature.name] = avg_distance
        
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
        rngs = {'dropout': jax.random.PRNGKey(0)}
        predictions = state.apply_fn({'params': state.params}, batch['inputs']['data'], batch['inputs']['blocks'], rngs=rngs)
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


def restore_train_state(checkpoint_dir, model, learning_rate, input_shape, block_shapes, rngs):
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=None)
    if restored_state:
        logger.info(f"Restored train state from {checkpoint_dir}")
        return restored_state
    else:
        logger.info("No checkpoint found. Initializing a new train state.")
        return create_train_state(rngs, model, learning_rate, input_shape, block_shapes)


def save_checkpoint(state, checkpoint_dir, prefix, epoch, max_to_keep=5):
    # Save the current state with epoch number
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=epoch,
        prefix=prefix,
        keep=max_to_keep
    )
    logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_dir}")

def main():
    config = ModelConfig()

    manager = TrackmaniaDataManager('trackmania_dataset.h5')
    map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'
    data_processor = DataProcessor(manager, map_uid, config)
    train_data, test_data, global_stats = data_processor.prepare_data()

    # Initialize Model
    model = BasicTrackmaniaNN(config=config)

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate=0.001, steps_per_epoch=len(train_data['inputs']['data']) // 64)

    # Create training state
    rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
    input_shape = train_data['inputs']['data'].shape
    block_shapes = {key: value.shape for key, value in train_data['inputs']['blocks'].items()}
    restore_state = False
    if restore_state:
        state = restore_train_state(
            checkpoint_dir=CHECKPOINT_DIR,
            model=model,
            learning_rate=learning_rate_fn,
            input_shape=input_shape,
            block_shapes=block_shapes,
            rngs=rngs
        )
    else:
        state = create_train_state(rngs, model, learning_rate=learning_rate_fn, input_shape=input_shape, block_shapes=block_shapes)

    batch_size = 64

    for epoch in range(state.step, config.num_epochs):
        batch_losses = []
        for batch in create_batches(train_data, batch_size):
            rng_key = jax.random.fold_in(rngs['dropout'], epoch * len(train_data['inputs']['data']) // batch_size + len(batch_losses))
            state, loss = train_step(state, batch, loss_weights, rng_key)
            batch_losses.append(loss)
        train_loss = jnp.mean(jnp.array(batch_losses))
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_accuracy(state, test_data)
            out = "  Accuracies: "
            for key, value in test_accuracy.items():
                out += f"'{key}': {value:.4f}, "
            print(out)
            predict_batch = create_batches(test_data, 32).__next__()
            pred, target = predict_single_batch(state, predict_batch)
            collect_and_save_predictions(pred, target, epoch + 1)
        
        if (epoch + 1) % 50 == 0:
            save_checkpoint(state, CHECKPOINT_DIR, CHECKPOINT_PREFIX, epoch + 1, CHECKPOINT_MAX_TO_KEEP)

if __name__ == '__main__':
    main()