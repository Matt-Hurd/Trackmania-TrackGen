import json
import numpy as np
import h5py

from enums import EventType
from features import Features

from collections import defaultdict


def tokenize_block(block, tokenizers):
    """Tokenize a single block based on provided tokenizers."""
    tokenized_block = {}
    for field in block.keys():
        if field in tokenizers:
            tokenized_block[field] = tokenizers[field].get(block[field], -1)  # Use -1 for unknown tokens
        else:
            tokenized_block[field] = block[field]
    return tokenized_block


class TrackmaniaDataManager:
    def __init__(self, filename):
        self.filename = filename
        self.file = h5py.File(filename, 'a')

        # Ensure 'maps' group exists
        if 'maps' not in self.file:
            self.file.create_group('maps')

    def add_map(self, map_data):
        map_uid = map_data['MapUid']
        map_path = f'maps/{map_uid}'

        if map_path in self.file:
            print(f"Map {map_uid} already exists. Skipping addition.")
            return self.file[map_path]

        # Create or get the map group
        map_group = self.file['maps'].require_group(map_uid)

        # Store map metadata
        for key in ['VehicleCollection', 'MapType', 'Author', 'MapStyle', 'MapName', 'MapUid']:
            map_group.attrs[key] = map_data.get(key, '')

        # Store block data
        blocks_group = map_group.require_group('blocks')
        for block_id, block_data in map_data.get('blocks', {}).items():
            block_group = blocks_group.require_group(block_id)
            for key, value in block_data.items():
                if isinstance(value, list):
                    # Overwrite existing dataset if it exists
                    if key in block_group:
                        del block_group[key]
                    block_group.create_dataset(key, data=np.array(value))
                else:
                    block_group.attrs[key] = value

        # Automatically generate tokenizers and tokenized blocks
        print(f"Generating tokenizers and tokenizing blocks for map {map_uid}.")
        tokenizers, tokenized_blocks = self.get_tokenizers(map_uid, block_hashes=None)
        self.save_tokenizers_and_blocks(map_uid, tokenizers, tokenized_blocks)

        return map_group

    def add_replay(self, map_uid, replay_data):
        map_path = f'maps/{map_uid}'
        if map_path not in self.file:
            print(f"Map {map_uid} does not exist. Cannot add replay.")
            return None

        map_group = self.file[map_path]
        replay_id = len([key for key in map_group.keys() if key.startswith('replay_')])
        replay_group_name = f'replay_{replay_id}'
        replay_group = map_group.require_group(replay_group_name)

        # Define the dtype for events
        events_dtype = np.dtype([
            ('Time', 'uint32'),
            ('Position', 'float32', (3,)),
            ('Left', 'float32', (3,)),
            ('Up', 'float32', (3,)),
            ('Dir', 'float32', (3,)),
            ('Velocity', 'float32', (3,)),
            ('WorldCarUp', 'float32', (3,)),
            ('CurGear', 'uint32'),
            ('FrontSpeed', 'float32'),
            ('InputSteer', 'float32'),
            ('InputGasPedal', 'float32'),
            ('InputBrakePedal', 'float32'),
            ('FLSteerAngle', 'float32'),
            ('FLWheelRot', 'float32'),
            ('FLWheelRotSpeed', 'float32'),
            ('FLDamperLen', 'float32'),
            ('FLSlipCoef', 'float32'),
            ('FRSteerAngle', 'float32'),
            ('FRWheelRot', 'float32'),
            ('FRWheelRotSpeed', 'float32'),
            ('FRDamperLen', 'float32'),
            ('FRSlipCoef', 'float32'),
            ('RLSteerAngle', 'float32'),
            ('RLWheelRot', 'float32'),
            ('RLWheelRotSpeed', 'float32'),
            ('RLDamperLen', 'float32'),
            ('RLSlipCoef', 'float32'),
            ('RRSteerAngle', 'float32'),
            ('RRWheelRot', 'float32'),
            ('RRWheelRotSpeed', 'float32'),
            ('RRDamperLen', 'float32'),
            ('RRSlipCoef', 'float32'),
            ('FLIcing01', 'float32'),
            ('FRIcing01', 'float32'),
            ('RLIcing01', 'float32'),
            ('RRIcing01', 'float32'),
            ('FLTireWear01', 'float32'),
            ('FRTireWear01', 'float32'),
            ('RLTireWear01', 'float32'),
            ('RRTireWear01', 'float32'),
            ('FLBreakNormedCoef', 'float32'),
            ('FRBreakNormedCoef', 'float32'),
            ('RLBreakNormedCoef', 'float32'),
            ('RRBreakNormedCoef', 'float32'),
            ('ReactorAirControl', 'float32', (3,)),
            ('GroundDist', 'float32'),
            ('ReactorInputsX', 'int32'),
            ('IsGroundContact', 'int32'),
            ('IsWheelsBurning', 'int32'),
            ('IsReactorGroundMode', 'int32'),
            ('EngineOn', 'int32'),
            ('IsTurbo', 'int32'),
            ('TurboTime', 'float32'),
            ('ReactorBoostType', 'int32'),
            ('ReactorBoostLvl', 'int32'),
            ('FLGroundContactMaterial', 'int32'),
            ('FRGroundContactMaterial', 'int32'),
            ('RLGroundContactMaterial', 'int32'),
            ('RRGroundContactMaterial', 'int32'),
            ('EventType', 'uint8'),
            ('BlockHash', h5py.string_dtype(encoding='utf-8')),
        ])

        # Check if 'events' dataset already exists; if so, delete and recreate
        if 'events' in replay_group:
            print(f"Replay {replay_group_name} already has 'events' dataset. Overwriting.")
            del replay_group['events']

        events_dataset = replay_group.create_dataset('events', (len(replay_data),), dtype=events_dtype)

        # Normalize time
        time_offset = replay_data[0]['Position'].get('Time', 0)

        # Fill the events dataset
        for i, frame in enumerate(replay_data):
            position = frame['Position']
            events_dataset[i] = (
                int(position['Time']) - int(time_offset),
                position.get('Position', [0.0, 0.0, 0.0]),
                position.get('Left', [0.0, 0.0, 0.0]),
                position.get('Up', [0.0, 0.0, 0.0]),
                position.get('Dir', [0.0, 0.0, 0.0]),
                position.get('WorldVel', [0.0, 0.0, 0.0]),
                position.get('WorldCarUp', [0.0, 0.0, 0.0]),
                int(position.get('CurGear', 0)),
                float(position.get('FrontSpeed', 0.0)),
                float(position.get('InputSteer', 0.0)),
                float(position.get('InputGasPedal', 0.0)),
                float(position.get('InputBrakePedal', 0.0)),
                float(position.get('FLSteerAngle', 0.0)),
                float(position.get('FLWheelRot', 0.0)),
                float(position.get('FLWheelRotSpeed', 0.0)),
                float(position.get('FLDamperLen', 0.0)),
                float(position.get('FLSlipCoef', 0.0)),
                float(position.get('FRSteerAngle', 0.0)),
                float(position.get('FRWheelRot', 0.0)),
                float(position.get('FRWheelRotSpeed', 0.0)),
                float(position.get('FRDamperLen', 0.0)),
                float(position.get('FRSlipCoef', 0.0)),
                float(position.get('RLSteerAngle', 0.0)),
                float(position.get('RLWheelRot', 0.0)),
                float(position.get('RLWheelRotSpeed', 0.0)),
                float(position.get('RLDamperLen', 0.0)),
                float(position.get('RLSlipCoef', 0.0)),
                float(position.get('RRSteerAngle', 0.0)),
                float(position.get('RRWheelRot', 0.0)),
                float(position.get('RRWheelRotSpeed', 0.0)),
                float(position.get('RRDamperLen', 0.0)),
                float(position.get('RRSlipCoef', 0.0)),
                float(position.get('FLIcing01', 0.0)),
                float(position.get('FRIcing01', 0.0)),
                float(position.get('RLIcing01', 0.0)),
                float(position.get('RRIcing01', 0.0)),
                float(position.get('FLTireWear01', 0.0)),
                float(position.get('FRTireWear01', 0.0)),
                float(position.get('RLTireWear01', 0.0)),
                float(position.get('RRTireWear01', 0.0)),
                float(position.get('FLBreakNormedCoef', 0.0)),
                float(position.get('FRBreakNormedCoef', 0.0)),
                float(position.get('RLBreakNormedCoef', 0.0)),
                float(position.get('RRBreakNormedCoef', 0.0)),
                position.get('ReactorAirControl', [0.0, 0.0, 0.0]),
                float(position.get('GroundDist', 0.0)),
                1 if position.get('ReactorInputsX', False) else 0,
                1 if position.get('IsGroundContact', False) else 0,
                1 if position.get('IsWheelsBurning', False) else 0,
                1 if position.get('IsReactorGroundMode', False) else 0,
                1 if position.get('EngineOn', False) else 0,
                1 if position.get('IsTurbo', False) else 0,
                float(position.get('TurboTime', 0.0)),
                int(position.get('ReactorBoostType', 0)),
                int(position.get('ReactorBoostLvl', 0)),
                int(position.get('FLGroundContactMaterial', 0)),
                int(position.get('FRGroundContactMaterial', 0)),
                int(position.get('RLGroundContactMaterial', 0)),
                int(position.get('RRGroundContactMaterial', 0)),
                frame.get('Type', 0),
                frame.get('BlockHash', ''),
            )

        return replay_group

    def load_map_data(self, map_file):
        with open(map_file, 'r') as f:
            map_data = json.load(f)
        return self.add_map(map_data)

    def load_replay_data(self, replay_file, map_uid):
        with open(replay_file, 'r') as f:
            replay_data = json.load(f)
        return self.add_replay(map_uid, replay_data)

    def close(self):
        self.file.close()

    def generate_training_examples(self, map_uid, sequence_length=32, stride=1):
        map_group = self.file[f'maps/{map_uid}']

        all_inputs = []
        all_targets = []

        for replay_name in [key for key in map_group.keys() if key.startswith('replay_')]:
            replay_group = map_group[replay_name]
            events = replay_group['events'][:]
            num_events = len(events)
            num_sequences = (num_events - sequence_length) // stride

            if num_sequences <= 0:
                continue

            # Generate start indices for sequences
            start_indices = np.arange(0, num_events - sequence_length, stride)

            # Generate indices for inputs and targets
            input_indices = start_indices[:, None] + np.arange(sequence_length)
            target_indices = input_indices + 1

            # Use advanced indexing to create sequences
            input_sequences = events[input_indices]
            target_sequences = events[target_indices]

            all_inputs.append(input_sequences)
            all_targets.append(target_sequences)

        # Concatenate all sequences from all replays
        if all_inputs:  # Check if the list is not empty
            all_inputs = np.concatenate(all_inputs, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_inputs = np.array([])
            all_targets = np.array([])

        return all_inputs, all_targets

    def prepare_data_for_training(self, map_uid, sequence_length=10, stride=1, test_split=0.2):
        inputs, targets = self.generate_training_examples(map_uid, sequence_length, stride)

        if len(inputs) == 0:
            print("No training data available.")
            return None, None

        # Shuffle the data
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = inputs[indices]
        targets = targets[indices]

        # Split into train and test sets
        split_index = int(len(inputs) * (1 - test_split))
        train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
        train_targets, test_targets = targets[:split_index], targets[split_index:]

        # Compute global_stats on the entire training set
        global_position_mean = np.mean(train_inputs[Features.POSITION.name].astype(np.float32), axis=(0, 1))
        global_position_std = np.std(train_inputs[Features.POSITION.name].astype(np.float32), axis=(0, 1))
        global_velocity_mean = np.mean(train_inputs[Features.VELOCITY.name].astype(np.float32), axis=(0, 1))
        global_velocity_std = np.std(train_inputs[Features.VELOCITY.name].astype(np.float32), axis=(0, 1))

        global_stats = (global_position_mean, global_position_std, global_velocity_mean, global_velocity_std)

        # Save global_stats
        self.save_global_stats(map_uid, global_stats)

        return {
            'train_inputs': train_inputs,
            'train_targets': train_targets,
            'test_inputs': test_inputs,
            'test_targets': test_targets
        }, global_stats

    def create_tokenizers(self, blocks, fields):
        """
        Create tokenizers for specified fields based on unique values in blocks.
        """
        tokenizers = {}
        for field in fields:
            unique_values = set()
            for block in blocks.values():
                unique_values.add(block[field])
            tokenizers[field] = {value: idx for idx, value in enumerate(sorted(unique_values))}
        return tokenizers

    def save_tokenizers_and_blocks(self, map_uid, tokenizers, tokenized_blocks):
        """
        Save tokenizers and tokenized blocks to the HDF5 file.
        """
        map_group = self.file[f'maps/{map_uid}']

        # Save tokenizers as attributes within 'tokenizers' group
        tokenizers_group = map_group.require_group('tokenizers')
        for field, tokenizer in tokenizers.items():
            # Overwrite the attribute if it already exists
            tokenizers_group.attrs[field] = json.dumps(tokenizer)

        # Save tokenized blocks as datasets within 'tokenized_blocks' group
        tokenized_blocks_group = map_group.require_group('tokenized_blocks')
        for block_id, block_data in tokenized_blocks.items():
            block_group = tokenized_blocks_group.require_group(block_id)
            for field, value in block_data.items():
                # If the dataset exists, delete it before creating a new one
                if field in block_group:
                    del block_group[field]
                block_group.create_dataset(field, data=np.array(value), dtype='int32')

    def load_tokenizers_and_blocks(self, map_uid, block_hashes):
        """
        Load tokenizers and tokenized blocks from the HDF5 file.
        """
        map_group = self.file[f'maps/{map_uid}']

        # Load tokenizers
        tokenizers_group = map_group['tokenizers']
        tokenizers = {field: json.loads(tokenizers_group.attrs[field]) for field in tokenizers_group.attrs}

        # Load tokenized blocks
        tokenized_blocks_group = map_group['tokenized_blocks']
        tokenized_blocks = {}
        for block_id in block_hashes:
            if block_id in tokenized_blocks_group:
                block_group = tokenized_blocks_group[block_id]
                block_data = {field: block_group[field][()].tolist() for field in block_group}
                tokenized_blocks[block_id] = block_data
            else:
                print(f"Warning: Tokenized block {block_id} not found in 'tokenized_blocks'.")
                # Optionally, handle missing blocks here

        return tokenizers, tokenized_blocks

    def get_tokenizers(self, map_uid, block_hashes=None):
        """
        Retrieve tokenizers and tokenized blocks. If block_hashes is None, retrieve all blocks.

        Args:
            map_uid (str): The unique identifier for the map.
            block_hashes (list or None): List of block hashes to retrieve. If None, retrieve all.

        Returns:
            Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int or original type]]]:
                - tokenizers for each field
                - tokenized blocks mapping block_id to tokenized data
        """
        map_group = self.file[f'maps/{map_uid}']

        # If tokenizers and tokenized_blocks already exist, load them
        if 'tokenizers' in map_group and 'tokenized_blocks' in map_group:
            print(f"Loading tokenizers and tokenized blocks from HDF5 for map {map_uid}.")
            if block_hashes is None:
                # Retrieve all block hashes
                block_hashes = list(map_group['tokenized_blocks'].keys())
            return self.load_tokenizers_and_blocks(map_uid, block_hashes)

        # Otherwise, generate the tokenizers and tokenized blocks as usual
        blocks_group = map_group['blocks']

        # If block_hashes is None, retrieve all block IDs
        if block_hashes is None:
            block_hashes = list(blocks_group.keys())

        blocks = {}
        for block_id in block_hashes:
            if block_id in blocks_group:
                block_data_group = blocks_group[block_id]
                blocks[block_id] = {
                    'BlockHash': block_id,
                    'BlockPosition': block_data_group['Position'][()].tolist(),
                    'BlockName': block_data_group.attrs.get('Name', ''),
                    'BlockInfoVariantIndex': int(block_data_group.attrs.get('BlockInfoVariantIndex', 0)),
                    'BlockDirection': block_data_group.attrs.get('Direction', ''),
                    'BlockPageName': block_data_group.attrs.get('PageName', ''),
                    'BlockMaterialName': block_data_group.attrs.get('MaterialName', ''),
                    'BlockCollectionId': block_data_group.attrs.get('CollectionId', '')
                }
            else:
                print(f"Warning: Block {block_id} not found in 'blocks'.")
                # Optionally, handle missing blocks here

        # Create tokenizers using only the loaded blocks
        fields_to_tokenize = ['BlockName', 'BlockHash', 'BlockPageName', 'BlockMaterialName', 'BlockCollectionId']
        tokenizers = self.create_tokenizers(blocks, fields_to_tokenize)

        # Tokenize all blocks
        tokenized_blocks = {block_id: tokenize_block(block_data, tokenizers) for block_id, block_data in blocks.items()}

        # Save the generated tokenizers and tokenized blocks into HDF5
        self.save_tokenizers_and_blocks(map_uid, tokenizers, tokenized_blocks)

        return tokenizers, tokenized_blocks

    def save_global_stats(self, map_uid, global_stats):
        """
        Save global statistics (mean and std) to the HDF5 file.
        """
        map_group = self.file[f'maps/{map_uid}']

        # Create or get 'global_stats' group
        global_stats_group = map_group.require_group('global_stats')

        # Overwrite datasets if they exist
        stats = ['position_mean', 'position_std', 'velocity_mean', 'velocity_std']
        for stat_name, stat_value in zip(stats, global_stats):
            if stat_name in global_stats_group:
                del global_stats_group[stat_name]
            global_stats_group.create_dataset(stat_name, data=stat_value)

    def load_global_stats(self, map_uid):
        """
        Load global statistics from the HDF5 file.
        """
        map_group = self.file[f'maps/{map_uid}']

        if 'global_stats' not in map_group:
            print("Global stats not found.")
            return None

        # Load the global stats if they exist
        global_stats_group = map_group['global_stats']
        global_position_mean = global_stats_group['position_mean'][()]
        global_position_std = global_stats_group['position_std'][()]
        global_velocity_mean = global_stats_group['velocity_mean'][()]
        global_velocity_std = global_stats_group['velocity_std'][()]

        return global_position_mean, global_position_std, global_velocity_mean, global_velocity_std

    def get_map_boundaries(self, map_uid):
        """
        Calculate and return the boundaries of the map based on block positions.
        """
        map_group = self.file[f'maps/{map_uid}']
        blocks_group = map_group['blocks']

        min_x = min_y = min_z = np.inf
        max_x = max_y = max_z = -np.inf

        for block_id, block_data in blocks_group.items():
            position = block_data['Position'][:]
            min_x = min(min_x, position[0])
            min_y = min(min_y, position[1])
            min_z = min(min_z, position[2])

            max_x = max(max_x, position[0])
            max_y = max(max_y, position[1])
            max_z = max(max_z, position[2])

        boundaries = {
            'MinX': min_x,
            'MinY': min_y,
            'MinZ': min_z,
            'MaxX': max_x,
            'MaxY': max_y,
            'MaxZ': max_z
        }

        print(boundaries)

        return boundaries

    def get_first_x_events_with_block_data(self, map_uid: str, replay_name: str, x: int):
        """
        Get the first `x` events for the given `replay_name` (including block data).

        Args:
            map_uid (str): The unique identifier for the map.
            replay_name (str): The replay to get events from.
            x (int): The number of events to retrieve.

        Returns:
            np.ndarray: Array of the first `x` events.
        """
        # Access the map group and replay group
        map_group = self.file[f'maps/{map_uid}']
        if replay_name not in map_group:
            print(f"Replay {replay_name} not found in map {map_uid}.")
            return None

        replay_group = map_group[replay_name]

        # Get all the events from the replay group and select the first `x` events
        events = replay_group['events'][:x]

        return events

    def list_all_maps(self):
        """
        Lists all map UIDs stored in the HDF5 file.
        """
        return list(self.file['maps'].keys())

    def list_all_replays(self, map_uid):
        """
        Lists all replays for a given map.

        Args:
            map_uid (str): The unique identifier for the map.

        Returns:
            list: A list of replay group names.
        """
        map_group = self.file[f'maps/{map_uid}']
        return [key for key in map_group.keys() if key.startswith('replay_')]