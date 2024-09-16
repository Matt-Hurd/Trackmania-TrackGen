from features import Features
import logging

loss_weights = {
    Features.POSITION.name: 0.01,
    Features.VELOCITY.name: 0.01,
    Features.EVENT_TYPE.name: 1,
    Features.TIME.name: 0.001,
    Features.BLOCK_NAME.name: 1.0,
    Features.BLOCK_PAGE_NAME.name: 1.0,
    Features.BLOCK_POSITION.name: 0.0001,
    Features.BLOCK_DIRECTION.name: 1,
    Features.INPUT_STEER.name: 0.01,
    Features.INPUT_GAS_PEDAL.name: 0.1,
    Features.INPUT_BRAKE_PEDAL.name: 0.1
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
