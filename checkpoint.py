import pathlib
import orbax.checkpoint as ocp
import os

CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)
CHECKPOINT_PREFIX = "train_state"
CHECKPOINT_MAX_TO_KEEP = 5

options = ocp.CheckpointManagerOptions(max_to_keep=CHECKPOINT_MAX_TO_KEEP)

def restore_train_state(train_state):
    with ocp.CheckpointManager(pathlib.Path(CHECKPOINT_DIR), options=options,) as mngr:
        latest_step = mngr.latest_step()
        if latest_step:
            return latest_step, mngr.restore(latest_step, args=ocp.args.StandardRestore(train_state))
    return 0, train_state

def save_checkpoint(state, epoch):
    with ocp.CheckpointManager(pathlib.Path(CHECKPOINT_DIR), options=options) as mngr:
        mngr.save(epoch, args=ocp.args.StandardSave(state))
