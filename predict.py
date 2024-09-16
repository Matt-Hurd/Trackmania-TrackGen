import os
import json
from typing import Dict, List
import jax
import numpy as np
import jax.numpy as jnp
from features import Features, EncodingType
from typing import Any

from config import loss_weights

PREDICT_DIR = './predictions'
os.makedirs(PREDICT_DIR, exist_ok=True)

def predict_single_batch(state, batch):    
    rngs = {'dropout': jax.random.PRNGKey(0)}
    predictions = state.apply_fn({'params': state.params}, batch['inputs']['data'], batch['inputs']['blocks'], rngs=rngs)
    targets = batch['targets']

    targets_dict = {
        'data': {},
        'blocks': {}
    }
    
    for feature in Features.get_all_features():
        if feature.encoding == EncodingType.NONE:
            continue
        if feature.is_block_feature:
            targets_dict['blocks'][feature.name] = targets['blocks'][feature.name]
        else:
            feature_slice = Features.get_feature_index(feature, False)
            if feature_slice is None:
                raise ValueError(f"No slice found for feature {feature.name}")
            true_feature = targets['data'][..., feature_slice]
            targets_dict['data'][feature.name] = true_feature

    return predictions, targets_dict

def collect_and_save_predictions(predictions: Dict[str, jnp.ndarray],
                                 targets: Dict[str, Any],
                                 epoch: int,
                                 max_save: int = 100):
    """
    Collects a subset of prediction-target pairs and saves them to disk.

    Args:
        predictions (Dict[str, jnp.ndarray]): Model predictions.
        targets (Dict[str, Any]): Ground truth targets.
        epoch (int): Current epoch number.
        max_save (int): Maximum number of samples to save per feature.
    """
    saved_data = {}
    
    for feature in Features.get_all_features():
        if feature.encoding == EncodingType.NONE or feature.name not in loss_weights:
            continue
        
        pred = predictions[feature.name]
        if feature.is_block_feature:
            true = targets['blocks'][feature.name]
        else:
            true = targets['data'][feature.name]
        
        if feature.encoding in [EncodingType.ONE_HOT, EncodingType.TOKENIZED]:
            pred_labels = jnp.argmax(pred, axis=-1)
            if feature.encoding == EncodingType.ONE_HOT:
                true_labels = jnp.argmax(true, axis=-1)
            else:
                true_labels = true  # Assuming true is already integer indices
            pred_labels = np.array(pred_labels[:max_save])
            true_labels = np.array(true_labels[:max_save])
            saved_data[feature.name] = {
                'predictions': pred_labels.tolist(),
                'targets': true_labels.tolist()
            }
        elif feature.encoding == EncodingType.NUMERICAL:
            pred_vals = np.array(pred[:max_save])
            true_vals = np.array(true[:max_save])
            saved_data[feature.name] = {
                'predictions': pred_vals.tolist(),
                'targets': true_vals.tolist()
            }
        else:
            # Handle other encodings if necessary
            pass
    
    filename = os.path.join(PREDICT_DIR, f'predictions_epoch_{epoch}.json')
    with open(filename, 'w') as f:
        json.dump(saved_data, f, indent=4)
    print(f"Saved prediction-target pairs to {filename}")
