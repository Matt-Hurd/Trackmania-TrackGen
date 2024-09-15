import json
import numpy as np
import h5py

class TrackmaniaDataManager:
    def __init__(self, filename):
        self.filename = filename
        self.file = h5py.File(filename, 'a')
        
        if 'maps' not in self.file:
            self.file.create_group('maps')

    def add_map(self, map_data):
        map_uid = map_data['MapUid']
        if f'maps/{map_uid}' in self.file:
            print(f"Map {map_uid} already exists. Skipping.")
            return self.file[f'maps/{map_uid}']

        map_group = self.file['maps'].create_group(map_uid)
        
        # Store map metadata
        for key in ['VehicleCollection', 'MapType', 'Author', 'MapStyle', 'MapName', 'MapUid']:
            map_group.attrs[key] = map_data[key]
        
        # Store block data
        blocks_group = map_group.create_group('blocks')
        for block_id, block_data in map_data['blocks'].items():
            block_group = blocks_group.create_group(block_id)
            for key, value in block_data.items():
                if isinstance(value, list):
                    block_group.create_dataset(key, data=np.array(value))
                else:
                    block_group.attrs[key] = value
        
        return map_group

    def add_replay(self, map_uid, replay_data):
        map_group = self.file[f'maps/{map_uid}']
        replay_id = len([key for key in map_group.keys() if key.startswith('replay_')])
        replay_group = map_group.create_group(f'replay_{replay_id}')

        # Create events dataset
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
            ('BlockHash', h5py.special_dtype(vlen=str)),
            ('EventType', 'uint8'),
            ('FLGroundContactMaterial', 'int32'),
            ('FRGroundContactMaterial', 'int32'),
            ('RLGroundContactMaterial', 'int32'),
            ('RRGroundContactMaterial', 'int32'),
            ('ReactorBoostType', 'int32'),
            ('ReactorBoostLvl', 'int32'),
        ])
        events_dataset = replay_group.create_dataset('events', (len(replay_data),), dtype=events_dtype)

        # Fill the events dataset
        for i, frame in enumerate(replay_data):
            position = frame['Position']
            events_dataset[i] = (
                int(position['Time']),
                position['Position'],
                position['Left'],
                position['Up'],
                position['Dir'],
                position['WorldVel'],
                position['WorldCarUp'],
                position['CurGear'],
                position['FrontSpeed'],
                position['InputSteer'],
                position['InputGasPedal'],
                position['InputBrakePedal'],
                position['FLSteerAngle'],
                position['FLWheelRot'],
                position['FLWheelRotSpeed'],
                position['FLDamperLen'],
                position['FLSlipCoef'],
                position['FRSteerAngle'],
                position['FRWheelRot'],
                position['FRWheelRotSpeed'],
                position['FRDamperLen'],
                position['FRSlipCoef'],
                position['RLSteerAngle'],
                position['RLWheelRot'],
                position['RLWheelRotSpeed'],
                position['RLDamperLen'],
                position['RLSlipCoef'],
                position['RRSteerAngle'],
                position['RRWheelRot'],
                position['RRWheelRotSpeed'],
                position['RRDamperLen'],
                position['RRSlipCoef'],
                position['FLIcing01'],
                position['FRIcing01'],
                position['RLIcing01'],
                position['RRIcing01'],
                position['FLTireWear01'],
                position['FRTireWear01'],
                position['RLTireWear01'],
                position['RRTireWear01'],
                position['FLBreakNormedCoef'],
                position['FRBreakNormedCoef'],
                position['RLBreakNormedCoef'],
                position['RRBreakNormedCoef'],
                position['ReactorAirControl'],
                position['GroundDist'],
                1 if position['ReactorInputsX'] else 0,
                1 if position['IsGroundContact'] else 0,
                1 if position['IsWheelsBurning'] else 0,
                1 if position['IsReactorGroundMode'] else 0,
                1 if position['EngineOn'] else 0,
                1 if position['IsTurbo'] else 0,
                position['TurboTime'],
                frame['BlockHash'],
                frame['Type'],
                position['FLGroundContactMaterial'],
                position['FRGroundContactMaterial'],
                position['RLGroundContactMaterial'],
                position['RRGroundContactMaterial'],
                position['ReactorBoostType'],
                position['ReactorBoostLvl'],
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

    def generate_training_examples(self, map_uid, sequence_length=10, stride=1):
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

        # Shuffle the data
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = inputs[indices]
        targets = targets[indices]

        # Split into train and test sets
        split_index = int(len(inputs) * (1 - test_split))
        train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
        train_targets, test_targets = targets[:split_index], targets[split_index:]

        return {
            'train_inputs': train_inputs,
            'train_targets': train_targets,
            'test_inputs': test_inputs,
            'test_targets': test_targets
        }

    def close(self):
        self.file.close()

    from collections import defaultdict

    def create_tokenizers(self, blocks, fields):
        tokenizers = {}
        for field in fields:
            unique_values = set()
            for block in blocks.values():
                unique_values.add(block[field])
            tokenizers[field] = {value: idx for idx, value in enumerate(unique_values)}
        return tokenizers

    def save_tokenizers_and_blocks(self, map_uid, tokenizers, tokenized_blocks):
        map_group = self.file[f'maps/{map_uid}']

        # Save tokenizers as attributes
        tokenizers_group = map_group.require_group('tokenizers')
        for field, tokenizer in tokenizers.items():
            tokenizer_data = json.dumps(tokenizer)
            tokenizers_group.attrs[field] = tokenizer_data  # Save tokenizer as a JSON string

        # Save tokenized blocks as datasets
        tokenized_blocks_group = map_group.require_group('tokenized_blocks')
        for block_id, block_data in tokenized_blocks.items():
            block_group = tokenized_blocks_group.create_group(block_id)
            for field, value in block_data.items():
                block_group.create_dataset(field, data=np.array(value))

    def load_tokenizers_and_blocks(self, map_uid, block_hashes):
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
                block_data = {field: block_group[field][()] for field in block_group}
                tokenized_blocks[block_id] = block_data
            else:
                print(f"Warning: Tokenized block {block_id} not found in tokenized_blocks_group.")
                # Handle missing blocks if necessary
        
        return tokenizers, tokenized_blocks

    def get_tokenizers(self, map_uid, block_hashes):
        # Check if tokenizers and tokenized blocks already exist in the HDF5 file
        map_group = self.file[f'maps/{map_uid}']
        if 'tokenizers' in map_group and 'tokenized_blocks' in map_group:
            print(f"Loading tokenizers and tokenized blocks from HDF5 for {map_uid}")
            return self.load_tokenizers_and_blocks(map_uid, block_hashes)

        # Otherwise, generate the tokenizers and tokenized blocks as usual
        blocks_group = map_group['blocks']
        blocks = {}
        for block_id in block_hashes:
            if block_id in blocks_group:
                block_data = blocks_group[block_id]
                blocks[block_id] = {
                    'BlockHash': block_id,
                    'Position': block_data['Position'][()],
                    'Name': block_data.attrs['Name'],
                    'BlockInfoVariantIndex': block_data.attrs['BlockInfoVariantIndex'],
                    'Direction': block_data.attrs['Direction'],
                    'PageName': block_data.attrs['PageName'],
                    'MaterialName': block_data.attrs['MaterialName'],
                    'CollectionId': block_data.attrs['CollectionId']
                }
            else:
                print(f"Warning: Block {block_id} not found in blocks_group.")
                # Handle missing blocks if necessary

        # Create tokenizers using only the loaded blocks
        tokenizers = self.create_tokenizers(blocks, ['Name', 'BlockHash', 'PageName', 'MaterialName', 'CollectionId'])
        tokenized_blocks = {block_id: tokenize_block(block_data, tokenizers) for block_id, block_data in blocks.items()}

        # Save the generated tokenizers and tokenized blocks into HDF5
        self.save_tokenizers_and_blocks(map_uid, tokenizers, tokenized_blocks)

        return tokenizers, tokenized_blocks

    def save_global_stats(self, map_uid, global_stats):
        map_group = self.file[f'maps/{map_uid}']

        # Create or overwrite 'global_stats' group
        global_stats_group = map_group.require_group('global_stats')

        # Save global stats (mean and std for position and velocity)
        global_stats_group.create_dataset('position_mean', data=global_stats[0])
        global_stats_group.create_dataset('position_std', data=global_stats[1])
        global_stats_group.create_dataset('velocity_mean', data=global_stats[2])
        global_stats_group.create_dataset('velocity_std', data=global_stats[3])

    def load_global_stats(self, map_uid):
        map_group = self.file[f'maps/{map_uid}']
        
        # Load the global stats if they exist
        global_stats_group = map_group['global_stats']
        global_position_mean = global_stats_group['position_mean'][()]
        global_position_std = global_stats_group['position_std'][()]
        global_velocity_mean = global_stats_group['velocity_mean'][()]
        global_velocity_std = global_stats_group['velocity_std'][()]

        return global_position_mean, global_position_std, global_velocity_mean, global_velocity_std

    
    def get_map_boundaries(self,map_uid):
        map_group = self.file[f'maps/{map_uid}']
        blocks_group = map_group['blocks']
        
        min_x = min_y = min_z = np.inf
        max_x = max_y = max_z = -np.inf
        
        for block_id, block_data in blocks_group.items():
            min_x = min(min_x, block_data['Position'][0])
            min_y = min(min_y, block_data['Position'][1])
            min_z = min(min_z, block_data['Position'][2])

            max_x = max(max_x, block_data['Position'][0])
            max_y = max(max_y, block_data['Position'][1])
            max_z = max(max_z, block_data['Position'][2])

        print({
            'MinX': min_x,
            'MinY': min_y,
            'MinZ': min_z,
            'MaxX': max_x,
            'MaxY': max_y,
            'MaxZ': max_z
        })
        
        return {
            'MinX': min_x,
            'MinY': min_y,
            'MinZ': min_z,
            'MaxX': max_x,
            'MaxY': max_y,
            'MaxZ': max_z
        }

def tokenize_block(block, tokenizers):
    tokenized_block = {}
    for field in block.keys():
        if field in tokenizers:
            tokenized_block[field] = tokenizers[field][block[field]]
        else:
            tokenized_block[field] = block[field]
    return tokenized_block