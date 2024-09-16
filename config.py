from features import Features

loss_weights = {
    Features.POSITION.name: 0.01,
    Features.VELOCITY.name: 0.01,
    Features.EVENT_TYPE.name: 1,
    Features.TIME.name: 0.001,
    Features.BLOCK_NAME.name: 1.0,
    Features.BLOCK_PAGE_NAME.name: 1.0,
    Features.BLOCK_POSITION.name: 0.01,
    Features.BLOCK_DIRECTION.name: 1,
    Features.INPUT_STEER.name: 1,
    Features.INPUT_GAS_PEDAL.name: 1,
    Features.INPUT_BRAKE_PEDAL.name: 1
}
