import glob
import logging
import pathlib
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.experimental import nnx
from flax.training import train_state
import optax
from typing import List, Sequence, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from checkpoint import restore_train_state, save_checkpoint
from enums import EncodingType, EventType
from features import Features

from config import loss_weights

from data_manager import TrackmaniaDataManager
from positional_encoding import PositionalEncoding
from predict import collect_and_save_predictions, predict_single_batch
from transformer_blocks import TransformerConfig, TransformerEncoderBlock

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
    num_epochs = 50000


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
    def __init__(self, manager: TrackmaniaDataManager, map_uid: str, config: ModelConfig, global_stats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None):
        self.manager = manager
        self.map_uid = map_uid
        self.config = config
        self.tokenized_blocks = None
        self.global_stats = global_stats
        self.tokenizers = None
    
    def load_all_tokenizers(self):
        self.tokenizers, self.tokenized_blocks = self.manager.get_tokenizers(self.map_uid)

    def prepare_data(self, mode='train') -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        raw_data, global_stats = self.manager.prepare_data_for_training(self.map_uid, sequence_length=32, stride=1, test_split=0.2)
        all_block_hashes = self.extract_all_block_hashes(raw_data)
        self.tokenizers, self.tokenized_blocks = self.manager.get_tokenizers(self.map_uid, all_block_hashes)

        # Update config sizes based on data
        self.update_config_sizes()

        if self.global_stats is None:
            # Load global_stats from DataManager
            self.global_stats = self.manager.load_global_stats(self.map_uid)

        # Normalize and preprocess
        train_data = self.preprocess_data(raw_data['train_inputs'])
        train_targets = self.preprocess_data(raw_data['train_targets'])
        train = {"inputs": train_data, "targets": train_targets}

        test_data = self.preprocess_data(raw_data['test_inputs'])
        test_targets = self.preprocess_data(raw_data['test_targets'])
        test = {"inputs": test_data, "targets": test_targets}

        return train, test

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

    def preprocess_data(self, data: np.ndarray) -> Dict[str, jnp.ndarray]:
        data = self.normalize_data(data)

        non_block_features = []
        for feature in Features.get_non_block_features():
            feature_data = data[feature.name].astype(np.float32)
            if feature.encoding == EncodingType.ONE_HOT:
                feature_data = jax.nn.one_hot(feature_data.astype(jnp.int32), feature.size)
            elif feature_data.ndim == 2:
                feature_data = feature_data.reshape(*feature_data.shape, 1)  # Add a feature dimension
            non_block_features.append(feature_data)

        # Concatenate non-block features
        x = jnp.concatenate(non_block_features, axis=-1)

        # Process block features
        block_data = self.process_block_features(data)

        return {"data": x, "blocks": block_data}

    def process_block_features(self, data: np.ndarray) -> Dict[str, jnp.ndarray]:
        event_types = data[Features.EVENT_TYPE.name].astype(jnp.int32)
        block_event_mask = jnp.isin(event_types, jnp.array([EventType.BLOCK_ENTER, EventType.BLOCK_EXIT]))
        block_indices = jnp.where(block_event_mask)
        block_hashes = data[Features.BLOCK_HASH.name][block_indices]

        block_field_values = get_block_data_for_hashes(block_hashes, self.tokenized_blocks)

        # Initialize block data with default values
        num_samples, num_timesteps = data.shape[:2]
        block_data = {}
        for feature in Features.get_block_features():
            default_value = get_default_block_values()[feature.name]
            if feature.encoding == EncodingType.ONE_HOT:
                block_data[feature.name] = jnp.full((num_samples, num_timesteps, feature.size), default_value, dtype=jnp.float32)
            elif feature.encoding == EncodingType.TOKENIZED:
                block_data[feature.name] = jnp.full((num_samples, num_timesteps), default_value, dtype=jnp.int32)
            else:
                block_data[feature.name] = jnp.full((num_samples, num_timesteps, feature.size), default_value, dtype=jnp.float32)

        # Assign block data using advanced indexing
        for idx in range(len(block_hashes)):
            sample_idx, timestep_idx = block_indices[0][idx], block_indices[1][idx]
            for feature in Features.get_block_features():
                if feature.encoding == EncodingType.NONE:
                    continue
                value = block_field_values[feature.name][idx]
                if feature.encoding == EncodingType.ONE_HOT:
                    if feature == Features.BLOCK_DIRECTION: # HACK
                        value = jax.nn.one_hot(value + 1, feature.size)
                    else:
                        value = jax.nn.one_hot(value, feature.size)
                block_data[feature.name] = block_data[feature.name].at[sample_idx, timestep_idx].set(value)

        return block_data

    def normalize_data(self, data: np.ndarray) -> None:
        global_position_mean, _, global_velocity_mean, _ = self.global_stats
        
        # Normalize time
        # Disabled for now
        # time = data[Features.TIME.name]
        # data[Features.TIME.name] = time - np.min(time, axis=1, keepdims=True)

        # Normalize position
        position = data[Features.POSITION.name]
        data[Features.POSITION.name] = (position - global_position_mean) # / (global_position_std + 1e-8)

        # Normalize velocity
        velocity = data[Features.VELOCITY.name]
        data[Features.VELOCITY.name] = (velocity - global_velocity_mean) # / (global_velocity_std + 1e-8)

class BasicTrackmaniaNN(nn.Module):
    config: ModelConfig
    max_seq_length: int = 32

    def setup(self):
        self.block_embeddings = {
            feature.name: nn.Embed(
                num_embeddings=feature.size,
                features=self.config.block_embedding_dim,
            )
            for feature in Features.get_block_features()
            if feature.encoding == EncodingType.TOKENIZED
        }
        
        self.input_projection = nn.Dense(self.config.d_model, use_bias=False)
        self.positional_encoding = PositionalEncoding(d_model=self.config.d_model)
        transformer_config = TransformerConfig(
            num_heads=self.config.num_heads,
            d_model=self.config.d_model,
            mlp_dim=self.config.mlp_dim,
            dropout_rate=self.config.dropout_rate,
            attention_dropout_rate=self.config.attention_dropout_rate,
            dtype=self.config.dtype,
            deterministic=self.config.deterministic,
        )
        self.transformer_layers = [
            TransformerEncoderBlock(config=transformer_config) for _ in range(self.config.num_layers)
        ]
        self.dense_layers = [nn.Dense(size) for size in self.config.hidden_sizes]
        self.output_layers =  {
            feature.name: nn.Dense(feature.size)
            for feature in Features.get_all_features()
            if feature.encoding != EncodingType.NONE
        }

    def __call__(self, x, block_data, train: bool = True):
        block_embeds = []
        for feature in Features.get_block_features():
            if feature.encoding == EncodingType.NONE:
                continue
            elif feature.encoding == EncodingType.TOKENIZED:
                block_embeds.append(self.block_embeddings[feature.name](block_data[feature.name]))
            elif feature.encoding == EncodingType.ONE_HOT:
                block_embeds.append(block_data[feature.name])
            else:
                block_embeds.append(block_data[feature.name])
        if block_embeds:
            x = jnp.concatenate([x, jnp.concatenate(block_embeds, axis=-1)], axis=-1)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x, train=train)
        for layer in self.dense_layers:
            x = nn.relu(layer(x))
        outputs = {}
        for feature_name, layer in self.output_layers.items():
            outputs[feature_name] = layer(x)
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

def create_train_state(rngs, model, learning_rate_fn, input_shape, block_shapes, batch_size):
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    dummy_block_data = {}
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

    model_vars = model.init(rngs, dummy_input, dummy_block_data)
    params = model_vars.get_only('params') # Get only 'params' part
    # assuming batch size is known and fixed, otherwise, you need to adjust this
    total_steps = model.config.num_epochs * (input_shape[0] // batch_size)
    learning_rate = learning_rate_fn(total_steps)  # Pass total_steps here
    tx = optax.chain(optax.adam(learning_rate), optax.clip_by_global_norm(1.0))
    opt_state = tx.init(params)
    return train_state.TrainState(params=params, opt_state=opt_state, tx=tx, step=0)

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
    event_types = targets['data'][..., Features.EVENT_TYPE.index]  # Get event types
    block_event_types = jnp.array([EventType.BLOCK_ENTER, EventType.BLOCK_EXIT])

    for feature in Features.get_all_features():
        if feature.encoding == EncodingType.NONE:
            continue

        pred = predictions[feature.name]

        if feature.is_block_feature:
            true = targets['blocks'][feature.name]
            block_event_mask = jnp.isin(event_types, block_event_types)
        else:
            true = targets['data'][..., feature.index]
            block_event_mask = jnp.ones_like(true, dtype=jnp.float32)

        if feature.encoding == EncodingType.ONE_HOT:
            loss = optax.softmax_cross_entropy(pred, true)
        elif feature.encoding == EncodingType.TOKENIZED:
            loss = optax.softmax_cross_entropy_with_integer_labels(pred, true)
        else:
            loss = (pred - true) ** 2

        masked_loss = jnp.mean(loss * block_event_mask)

        total_loss += loss_weights.get(feature.name, 0.1) * masked_loss

    return total_loss

@jax.jit
def train_step(state: train_state.TrainState, model: BasicTrackmaniaNN, batch, loss_weights, rng_key):
    def loss_fn(params):
        rngs = {'dropout': rng_key}
        predictions = model.apply(
            {'params': params},
            batch['inputs']['data'],
            batch['inputs']['blocks'],
            rngs=rngs,
            train=True,
        )
        return custom_loss(predictions, batch['targets'], loss_weights)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = train_state.TrainState(params=new_params, opt_state=new_opt_state, tx=state.tx, step=state.step + 1)
    return new_state, loss


@jax.jit
def eval_step(state: train_state.TrainState, model: BasicTrackmaniaNN, batch, loss_weights, rng_key):
    rngs = {'dropout': rng_key}
    predictions = model.apply(
        {'params': state.params},
        batch['inputs']['data'],
        batch['inputs']['blocks'],
        rngs=rngs,
        train=False,
    )
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
            true = targets['data'][..., feature.index]

        if feature.encoding == EncodingType.ONE_HOT:
            pred_labels = jnp.argmax(pred, axis=-1)
            true_labels = jnp.argmax(true, axis=-1)
            acc = jnp.mean(pred_labels == true_labels)
            accuracies[feature.name] = acc
        elif feature.encoding == EncodingType.TOKENIZED:
            pred_labels = jnp.argmax(pred, axis=-1)
            true_labels = true
            acc = jnp.mean(pred_labels == true_labels)
            accuracies[feature.name] = acc
        elif feature.encoding == EncodingType.NUMERICAL:
            distances = jnp.abs(pred - true)
            avg_distance = jnp.mean(distances)
            accuracies[feature.name] = avg_distance

    return accuracies

@jax.jit
def evaluate_accuracy(state: train_state.TrainState, model: BasicTrackmaniaNN, data, batch_size=32):
    all_predictions = []
    all_targets = []

    for batch in create_batches(data, batch_size):
        rngs = {'dropout': jax.random.PRNGKey(0)}
        predictions = model.apply(
            {'params': state.params},
            batch['inputs']['data'],
            batch['inputs']['blocks'],
            rngs=rngs,
            train=False,
        )
        all_predictions.append(predictions)
        all_targets.append(batch['targets'])

    combined_predictions = {key: [] for key in all_predictions[0]}
    combined_targets = {
        'data': [],
        'blocks': {feature.name: [] for feature in Features.get_block_features()}
    }

    for preds, targs in zip(all_predictions, all_targets):
        for key in preds:
            combined_predictions[key].append(preds[key])
        
        combined_targets['data'].append(targs['data'])
        for feature in Features.get_block_features():
            combined_targets['blocks'][feature.name].append(targs['blocks'][feature.name])

    for key in combined_predictions:
        combined_predictions[key] = jnp.concatenate(combined_predictions[key], axis=0)

    combined_targets['data'] = jnp.concatenate(combined_targets['data'], axis=0)
    for feature in Features.get_block_features():
        combined_targets['blocks'][feature.name] = jnp.concatenate(combined_targets['blocks'][feature.name], axis=0)

    accuracies = calculate_accuracy(combined_predictions, combined_targets)

    return accuracies

def main():
    config = ModelConfig()

    manager = TrackmaniaDataManager('trackmania_dataset.h5')
    map_uid = 'DUzLndlMvwhFmzDkp4JSQFuuj1b'
    data_processor = DataProcessor(manager, map_uid)
    train_data, test_data = data_processor.prepare_data()

    # Initialize Model
    model = BasicTrackmaniaNN(config=config)

    batch_size = 64
    learning_rate_fn = create_learning_rate_fn(
        config, base_learning_rate=0.0001, steps_per_epoch=len(train_data['inputs']['data']) // batch_size
    )

    # Create training state
    rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
    input_shape = train_data['inputs']['data'].shape
    block_shapes = {key: value.shape for key, value in train_data['inputs']['blocks'].items()}

    restore_checkpoint = False
    if restore_checkpoint:
        state = restore_train_state("checkpoints/checkpoint_epoch_50.pkl") # Example path
        start_epoch = state.step // (len(train_data['inputs']['data']) // batch_size)
    else:
        state = create_train_state(rngs, model, learning_rate_fn, input_shape, block_shapes, batch_size)
        start_epoch = 0

    for epoch in range(start_epoch, config.num_epochs):
        batch_losses = []
        for batch in create_batches(train_data, batch_size):
            rng_key = jax.random.fold_in(rngs['dropout'], epoch * len(train_data['inputs']['data']) // batch_size + len(batch_losses))
            state, loss = train_step(state, model, batch, loss_weights, rng_key)
            batch_losses.append(loss)

        train_loss = jnp.mean(jnp.array(batch_losses))
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_accuracy(state, model, test_data)
            out = "  Accuracies: "
            for key, value in test_accuracy.items():
                out += f"'{key}': {value:.4f}, "
            print(out)

            predict_batch = next(create_batches(test_data, 32))
            pred, target = predict_single_batch(state, model, predict_batch)
            collect_and_save_predictions(pred, target, epoch + 1)

            save_checkpoint(state, epoch + 1)

if __name__ == '__main__':
    main()