from data_manager import TrackmaniaDataManager

manager = TrackmaniaDataManager('trackmania_dataset.h5')

def create_dataset():
    map_group = manager.load_map_data('Dumps/map.json')
    replay_group = manager.load_replay_data('Dumps/Ghost-Summer 2024 - 02 - Kelven..json', map_group.name.split('/')[-1])
    replay_group = manager.load_replay_data('Dumps/Ghost-Summer 2024 -02 - R1nt3..json', map_group.name.split('/')[-1])
    replay_group = manager.load_replay_data('Dumps/Ghost-Summer 2024 -02 - zodivagantz.json', map_group.name.split('/')[-1])

    manager.close()

create_dataset()