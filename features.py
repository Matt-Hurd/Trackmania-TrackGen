import flax.struct
import jax.numpy as jnp
from typing import Tuple
from enums import Material, ReactorBoostType, ReactorBoostLevel, EventType, BlockDirection, EncodingType

@flax.struct.dataclass
class Feature:
    name: str
    size: int
    encoding: EncodingType
    is_block_feature: bool = False
    index: int = flax.struct.field(default=-1, pytree_node=False)

class Features:
    # Non-block features
    TIME = Feature("Time", 1, EncodingType.NUMERICAL, index=0)
    POSITION = Feature("Position", 3, EncodingType.NUMERICAL, index=1)
    LEFT = Feature("Left", 3, EncodingType.NUMERICAL, index=2)
    UP = Feature("Up", 3, EncodingType.NUMERICAL, index=3)
    DIR = Feature("Dir", 3, EncodingType.NUMERICAL, index=4)
    VELOCITY = Feature("Velocity", 3, EncodingType.NUMERICAL, index=5)
    WORLD_CAR_UP = Feature("WorldCarUp", 3, EncodingType.NUMERICAL, index=6)
    FRONT_SPEED = Feature("FrontSpeed", 1, EncodingType.NUMERICAL, index=7)
    INPUT_STEER = Feature("InputSteer", 1, EncodingType.NUMERICAL, index=8)
    FL_STEER_ANGLE = Feature("FLSteerAngle", 1, EncodingType.NUMERICAL, index=9)
    FL_WHEEL_ROT = Feature("FLWheelRot", 1, EncodingType.NUMERICAL, index=10)
    FL_WHEEL_ROT_SPEED = Feature("FLWheelRotSpeed", 1, EncodingType.NUMERICAL, index=11)
    FL_DAMPER_LEN = Feature("FLDamperLen", 1, EncodingType.NUMERICAL, index=12)
    FL_SLIP_COEF = Feature("FLSlipCoef", 1, EncodingType.NUMERICAL, index=13)
    FR_STEER_ANGLE = Feature("FRSteerAngle", 1, EncodingType.NUMERICAL, index=14)
    FR_WHEEL_ROT = Feature("FRWheelRot", 1, EncodingType.NUMERICAL, index=15)
    FR_WHEEL_ROT_SPEED = Feature("FRWheelRotSpeed", 1, EncodingType.NUMERICAL, index=16)
    FR_DAMPER_LEN = Feature("FRDamperLen", 1, EncodingType.NUMERICAL, index=17)
    FR_SLIP_COEF = Feature("FRSlipCoef", 1, EncodingType.NUMERICAL, index=18)
    RL_STEER_ANGLE = Feature("RLSteerAngle", 1, EncodingType.NUMERICAL, index=19)
    RL_WHEEL_ROT = Feature("RLWheelRot", 1, EncodingType.NUMERICAL, index=20)
    RL_WHEEL_ROT_SPEED = Feature("RLWheelRotSpeed", 1, EncodingType.NUMERICAL, index=21)
    RL_DAMPER_LEN = Feature("RLDamperLen", 1, EncodingType.NUMERICAL, index=22)
    RL_SLIP_COEF = Feature("RLSlipCoef", 1, EncodingType.NUMERICAL, index=23)
    RR_STEER_ANGLE = Feature("RRSteerAngle", 1, EncodingType.NUMERICAL, index=24)
    RR_WHEEL_ROT = Feature("RRWheelRot", 1, EncodingType.NUMERICAL, index=25)
    RR_WHEEL_ROT_SPEED = Feature("RRWheelRotSpeed", 1, EncodingType.NUMERICAL, index=26)
    RR_DAMPER_LEN = Feature("RRDamperLen", 1, EncodingType.NUMERICAL, index=27)
    RR_SLIP_COEF = Feature("RRSlipCoef", 1, EncodingType.NUMERICAL, index=28)
    FL_ICING = Feature("FLIcing01", 1, EncodingType.NUMERICAL, index=29)
    FR_ICING = Feature("FRIcing01", 1, EncodingType.NUMERICAL, index=30)
    RL_ICING = Feature("RLIcing01", 1, EncodingType.NUMERICAL, index=31)
    RR_ICING = Feature("RRIcing01", 1, EncodingType.NUMERICAL, index=32)
    FL_TIRE_WEAR = Feature("FLTireWear01", 1, EncodingType.NUMERICAL, index=33)
    FR_TIRE_WEAR = Feature("FRTireWear01", 1, EncodingType.NUMERICAL, index=34)
    RL_TIRE_WEAR = Feature("RLTireWear01", 1, EncodingType.NUMERICAL, index=35)
    RR_TIRE_WEAR = Feature("RRTireWear01", 1, EncodingType.NUMERICAL, index=36)
    FL_BREAK_NORMED_COEF = Feature("FLBreakNormedCoef", 1, EncodingType.NUMERICAL, index=37)
    FR_BREAK_NORMED_COEF = Feature("FRBreakNormedCoef", 1, EncodingType.NUMERICAL, index=38)
    RL_BREAK_NORMED_COEF = Feature("RLBreakNormedCoef", 1, EncodingType.NUMERICAL, index=39)
    RR_BREAK_NORMED_COEF = Feature("RRBreakNormedCoef", 1, EncodingType.NUMERICAL, index=40)
    REACTOR_AIR_CONTROL = Feature("ReactorAirControl", 3, EncodingType.NUMERICAL, index=41)
    GROUND_DIST = Feature("GroundDist", 1, EncodingType.NUMERICAL, index=42)
    TURBO_TIME = Feature("TurboTime", 1, EncodingType.NUMERICAL, index=43)

    REACTOR_INPUTS_X = Feature("ReactorInputsX", 2, EncodingType.ONE_HOT, index=44)
    IS_GROUND_CONTACT = Feature("IsGroundContact", 2, EncodingType.ONE_HOT, index=45)
    IS_WHEELS_BURNING = Feature("IsWheelsBurning", 2, EncodingType.ONE_HOT, index=46)
    IS_REACTOR_GROUND_MODE = Feature("IsReactorGroundMode", 2, EncodingType.ONE_HOT, index=47)
    INPUT_GAS_PEDAL = Feature("InputGasPedal", 2, EncodingType.ONE_HOT, index=48)
    INPUT_BRAKE_PEDAL = Feature("InputBrakePedal", 2, EncodingType.ONE_HOT, index=49)
    ENGINE_ON = Feature("EngineOn", 2, EncodingType.ONE_HOT, index=50)
    IS_TURBO = Feature("IsTurbo", 2, EncodingType.ONE_HOT, index=51)

    REACTOR_BOOST_TYPE = Feature("ReactorBoostType", len(ReactorBoostType), EncodingType.ONE_HOT, index=52)
    REACTOR_BOOST_LVL = Feature("ReactorBoostLvl", len(ReactorBoostLevel), EncodingType.ONE_HOT, index=53)
    FL_GROUND_CONTACT_MATERIAL = Feature("FLGroundContactMaterial", len(Material), EncodingType.ONE_HOT, index=54)
    FR_GROUND_CONTACT_MATERIAL = Feature("FRGroundContactMaterial", len(Material), EncodingType.ONE_HOT, index=55)
    RL_GROUND_CONTACT_MATERIAL = Feature("RLGroundContactMaterial", len(Material), EncodingType.ONE_HOT, index=56)
    RR_GROUND_CONTACT_MATERIAL = Feature("RRGroundContactMaterial", len(Material), EncodingType.ONE_HOT, index=57)
    CUR_GEAR = Feature("CurGear", 6, EncodingType.ONE_HOT, index=58)

    EVENT_TYPE = Feature("EventType", len(EventType), EncodingType.ONE_HOT, index=59)
    BLOCK_HASH = Feature("BlockHash", 0, EncodingType.NONE, is_block_feature=True)  # size will be set dynamically

    # Block features
    BLOCK_POSITION = Feature("BlockPosition", 3, EncodingType.NUMERICAL, is_block_feature=True)
    BLOCK_DIRECTION = Feature("BlockDirection", len(BlockDirection), EncodingType.ONE_HOT, is_block_feature=True)
    BLOCK_NAME = Feature("BlockName", 0, EncodingType.TOKENIZED, is_block_feature=True)  # size will be set dynamically
    BLOCK_PAGE_NAME = Feature("BlockPageName", 0, EncodingType.TOKENIZED, is_block_feature=True)  # size will be set dynamically
    BLOCK_MATERIAL_NAME = Feature("BlockMaterialName", 0, EncodingType.TOKENIZED, is_block_feature=True)  # size will be set dynamically

    @classmethod
    def get_all_features(cls) -> Tuple[Feature, ...]:
        return tuple(v for k, v in cls.__dict__.items() if isinstance(v, Feature))

    @classmethod
    def get_non_block_features(cls) -> Tuple[Feature, ...]:
        return tuple(f for f in cls.get_all_features() if not f.is_block_feature)

    @classmethod
    def get_block_features(cls) -> Tuple[Feature, ...]:
        return tuple(f for f in cls.get_all_features() if f.is_block_feature)
    
    @classmethod
    def get_numerical_features(cls) -> Tuple[Feature, ...]:
        return tuple(f for f in cls.get_all_features() if f.encoding == EncodingType.NUMERICAL)

    @classmethod
    def set_feature_size(cls, feature: Feature, size: int):
        if feature.name == "BlockDirection":
            setattr(cls, feature.name, feature.replace(size=7)) # HACK
        else:
            setattr(cls, feature.name, feature.replace(size=size))

    @classmethod
    def get_feature_index(cls, feature: Feature, input: bool = True) -> int:
        # For simplicity, we're just returning the pre-assigned index
        # You can add more complex logic here if needed
        return feature.index

    @classmethod
    def get_feature_slices(cls):
        # This assumes each feature occupies a contiguous block of indices
        slices = {}
        for feature in cls.get_all_features():
            if feature.is_block_feature:
                continue
            start = cls.get_feature_index(feature)
            end = start + feature.size
            slices[feature.name] = slice(start, end)
        return slices