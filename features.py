from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enums import Material, ReactorBoostType, ReactorBoostLevel, EventType, BlockDirection, EncodingType

@dataclass
class FeatureInfo:
    name: str
    size: int
    encoding: EncodingType
    vocabulary_size: Optional[int] = None
    is_block_feature: bool = False
    _output_index: slice = None
    _input_index: slice = None

    def __hash__(self):
        return hash((self.name, self.encoding, self.is_block_feature))

class OrderedFeaturesMeta(type):
    def __new__(cls, name, bases, attrs):
        features = []
        for key, value in attrs.items():
            if isinstance(value, FeatureInfo):
                features.append((key, value))
        attrs['_features'] = [f[1] for f in features]
        return super().__new__(cls, name, bases, attrs)

    def __iter__(self):
        return iter(self._features)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._features[key]
        for feature in self._features:
            if feature.name == key:
                return feature
        raise KeyError(key)

class Features(metaclass=OrderedFeaturesMeta):
    TIME                        = FeatureInfo("Time", 1, EncodingType.NUMERICAL)
    POSITION                    = FeatureInfo("Position", 3, EncodingType.NUMERICAL)
    LEFT                        = FeatureInfo("Left", 3, EncodingType.NUMERICAL)
    UP                          = FeatureInfo("Up", 3, EncodingType.NUMERICAL)
    DIR                         = FeatureInfo("Dir", 3, EncodingType.NUMERICAL)
    VELOCITY                    = FeatureInfo("Velocity", 3, EncodingType.NUMERICAL)
    WORLD_CAR_UP                = FeatureInfo("WorldCarUp", 3, EncodingType.NUMERICAL)
    IS_GROUND_CONTACT           = FeatureInfo("IsGroundContact", 1, EncodingType.NUMERICAL)
    IS_WHEELS_BURNING           = FeatureInfo("IsWheelsBurning", 1, EncodingType.NUMERICAL)
    IS_REACTOR_GROUND_MODE      = FeatureInfo("IsReactorGroundMode", 1, EncodingType.NUMERICAL)
    CUR_GEAR                    = FeatureInfo("CurGear", 1, EncodingType.NUMERICAL)
    FRONT_SPEED                 = FeatureInfo("FrontSpeed", 1, EncodingType.NUMERICAL)
    INPUT_STEER                 = FeatureInfo("InputSteer", 1, EncodingType.NUMERICAL)
    INPUT_GAS_PEDAL             = FeatureInfo("InputGasPedal", 1, EncodingType.NUMERICAL)
    INPUT_BRAKE_PEDAL           = FeatureInfo("InputBrakePedal", 1, EncodingType.NUMERICAL)
    FL_STEER_ANGLE              = FeatureInfo("FLSteerAngle", 1, EncodingType.NUMERICAL)
    FL_WHEEL_ROT                = FeatureInfo("FLWheelRot", 1, EncodingType.NUMERICAL)
    FL_WHEEL_ROT_SPEED          = FeatureInfo("FLWheelRotSpeed", 1, EncodingType.NUMERICAL)
    FL_DAMPER_LEN               = FeatureInfo("FLDamperLen", 1, EncodingType.NUMERICAL)
    FL_SLIP_COEF                = FeatureInfo("FLSlipCoef", 1, EncodingType.NUMERICAL)
    FR_STEER_ANGLE              = FeatureInfo("FRSteerAngle", 1, EncodingType.NUMERICAL)
    FR_WHEEL_ROT                = FeatureInfo("FRWheelRot", 1, EncodingType.NUMERICAL)
    FR_WHEEL_ROT_SPEED          = FeatureInfo("FRWheelRotSpeed", 1, EncodingType.NUMERICAL)
    FR_DAMPER_LEN               = FeatureInfo("FRDamperLen", 1, EncodingType.NUMERICAL)
    FR_SLIP_COEF                = FeatureInfo("FRSlipCoef", 1, EncodingType.NUMERICAL)
    RL_STEER_ANGLE              = FeatureInfo("RLSteerAngle", 1, EncodingType.NUMERICAL)
    RL_WHEEL_ROT                = FeatureInfo("RLWheelRot", 1, EncodingType.NUMERICAL)
    RL_WHEEL_ROT_SPEED          = FeatureInfo("RLWheelRotSpeed", 1, EncodingType.NUMERICAL)
    RL_DAMPER_LEN               = FeatureInfo("RLDamperLen", 1, EncodingType.NUMERICAL)
    RL_SLIP_COEF                = FeatureInfo("RLSlipCoef", 1, EncodingType.NUMERICAL)
    RR_STEER_ANGLE              = FeatureInfo("RRSteerAngle", 1, EncodingType.NUMERICAL)
    RR_WHEEL_ROT                = FeatureInfo("RRWheelRot", 1, EncodingType.NUMERICAL)
    RR_WHEEL_ROT_SPEED          = FeatureInfo("RRWheelRotSpeed", 1, EncodingType.NUMERICAL)
    RR_DAMPER_LEN               = FeatureInfo("RRDamperLen", 1, EncodingType.NUMERICAL)
    RR_SLIP_COEF                = FeatureInfo("RRSlipCoef", 1, EncodingType.NUMERICAL)
    FL_ICING                    = FeatureInfo("FLIcing01", 1, EncodingType.NUMERICAL)
    FR_ICING                    = FeatureInfo("FRIcing01", 1, EncodingType.NUMERICAL)
    RL_ICING                    = FeatureInfo("RLIcing01", 1, EncodingType.NUMERICAL)
    RR_ICING                    = FeatureInfo("RRIcing01", 1, EncodingType.NUMERICAL)
    FL_TIRE_WEAR                = FeatureInfo("FLTireWear01", 1, EncodingType.NUMERICAL)
    FR_TIRE_WEAR                = FeatureInfo("FRTireWear01", 1, EncodingType.NUMERICAL)
    RL_TIRE_WEAR                = FeatureInfo("RLTireWear01", 1, EncodingType.NUMERICAL)
    RR_TIRE_WEAR                = FeatureInfo("RRTireWear01", 1, EncodingType.NUMERICAL)
    FL_BREAK_NORMED_COEF        = FeatureInfo("FLBreakNormedCoef", 1, EncodingType.NUMERICAL)
    FR_BREAK_NORMED_COEF        = FeatureInfo("FRBreakNormedCoef", 1, EncodingType.NUMERICAL)
    RL_BREAK_NORMED_COEF        = FeatureInfo("RLBreakNormedCoef", 1, EncodingType.NUMERICAL)
    RR_BREAK_NORMED_COEF        = FeatureInfo("RRBreakNormedCoef", 1, EncodingType.NUMERICAL)
    REACTOR_AIR_CONTROL         = FeatureInfo("ReactorAirControl", 3, EncodingType.NUMERICAL)
    REACTOR_INPUTS_X            = FeatureInfo("ReactorInputsX", 1, EncodingType.NUMERICAL)
    GROUND_DIST                 = FeatureInfo("GroundDist", 1, EncodingType.NUMERICAL)
    ENGINE_ON                   = FeatureInfo("EngineOn", 1, EncodingType.NUMERICAL)
    IS_TURBO                    = FeatureInfo("IsTurbo", 1, EncodingType.NUMERICAL)
    TURBO_TIME                  = FeatureInfo("TurboTime", 1, EncodingType.NUMERICAL)
    REACTOR_BOOST_TYPE          = FeatureInfo("ReactorBoostType", len(ReactorBoostType), EncodingType.ONE_HOT)
    REACTOR_BOOST_LVL           = FeatureInfo("ReactorBoostLvl", len(ReactorBoostLevel), EncodingType.ONE_HOT)
    FL_GROUND_CONTACT_MATERIAL  = FeatureInfo("FLGroundContactMaterial", len(Material), EncodingType.ONE_HOT)
    FR_GROUND_CONTACT_MATERIAL  = FeatureInfo("FRGroundContactMaterial", len(Material), EncodingType.ONE_HOT)
    RL_GROUND_CONTACT_MATERIAL  = FeatureInfo("RLGroundContactMaterial", len(Material), EncodingType.ONE_HOT)
    RR_GROUND_CONTACT_MATERIAL  = FeatureInfo("RRGroundContactMaterial", len(Material), EncodingType.ONE_HOT)
    EVENT_TYPE                  = FeatureInfo("EventType", len(EventType), EncodingType.ONE_HOT)
    BLOCK_HASH                  = FeatureInfo("BlockHash", 0, EncodingType.NONE, is_block_feature=True)  # size will be set dynamically

    BLOCK_POSITION      = FeatureInfo("BlockPosition", 3, EncodingType.NUMERICAL, is_block_feature=True)
    BLOCK_DIRECTION     = FeatureInfo("BlockDirection", len(BlockDirection), EncodingType.ONE_HOT, is_block_feature=True)
    BLOCK_NAME          = FeatureInfo("BlockName", None, EncodingType.TOKENIZED, is_block_feature=True)  # size will be set dynamically
    BLOCK_PAGE_NAME     = FeatureInfo("BlockPageName", None, EncodingType.TOKENIZED, is_block_feature=True)  # size will be set dynamically
    BLOCK_MATERIAL_NAME = FeatureInfo("BlockMaterialName", None, EncodingType.TOKENIZED, is_block_feature=True)  # size will be set dynamically

    _needs_reindexing = True

    @classmethod
    def get_feature_index(cls, feature: FeatureInfo, input: bool=True) -> int:
        if cls._needs_reindexing:
            cls._reindex_features()
        return feature._input_index if input else feature._output_index
    
    @classmethod
    def set_feature_size(cls, feature: FeatureInfo, size: int):
        if feature.size is not None:
            raise ValueError(f"Feature {feature.name} already has a size.")
        feature.size = size
        cls._needs_reindexing = True

    @classmethod
    def _reindex_features(cls):
        input_idx = 0
        output_idx = 0
        for feature in cls.get_all_features():
            if feature.size is None:
                raise ValueError(f"Feature {feature.name} does not have a size yet.")
            feature_size = 1 if feature.encoding in [EncodingType.ONE_HOT, EncodingType.TOKENIZED] else feature.size
            feature._input_index = slice(input_idx, input_idx + feature_size)
            input_idx += feature_size
            
            feature_size = 1 if feature.encoding in [EncodingType.TOKENIZED] else feature.size
            feature._output_index = slice(output_idx, output_idx + feature_size)
            output_idx += feature_size
        cls._needs_reindexing = False

    @classmethod
    def get_all_features(cls) -> List[FeatureInfo]:
        return cls._features.copy()

    @classmethod
    def get_block_features(cls) -> List[FeatureInfo]:
        return [feature for feature in cls.get_all_features() if feature.is_block_feature]

    @classmethod
    def get_numerical_features(cls) -> List[FeatureInfo]:
         return [feature for feature in cls.get_all_features() if feature.encoding == EncodingType.NUMERICAL]
    
    @staticmethod
    def get_feature_slices():
        slices = {}
        current = 0
        for feature in Features.get_all_features():
            if feature.is_block_feature:
                continue
            if feature.encoding in [EncodingType.ONE_HOT, EncodingType.NUMERICAL]:
                start = current
                end = current + feature.size
                slices[feature.name] = slice(start, end)
                current = end
            elif feature.encoding == EncodingType.TOKENIZED:
                start = current
                end = current + 1
                slices[feature.name] = slice(start, end)
                current = end
        return slices
