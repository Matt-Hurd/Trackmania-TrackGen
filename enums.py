from enum import Enum, auto, IntEnum

class EncodingType(Enum):
    NUMERICAL = auto()
    ONE_HOT = auto()
    TOKENIZED = auto()
    NONE = auto()

class Material(Enum):
    Concrete = auto()
    Pavement = auto()
    Grass = auto()
    Ice = auto()
    Metal = auto()
    Sand = auto()
    Dirt = auto()
    Turbo_Deprecated = auto()
    DirtRoad = auto()
    Rubber = auto()
    SlidingRubber = auto()
    Test = auto()
    Rock = auto()
    Water = auto()
    Wood = auto()
    Danger = auto()
    Asphalt = auto()
    WetDirtRoad = auto()
    WetAsphalt = auto()
    WetPavement = auto()
    WetGrass = auto()
    Snow = auto()
    ResonantMetal = auto()
    GolfBall = auto()
    GolfWall = auto()
    GolfGround = auto()
    Turbo2_Deprecated = auto()
    Bumper_Deprecated = auto()
    NotCollidable = auto()
    FreeWheeling_Deprecated = auto()
    TurboRoulette_Deprecated = auto()
    WallJump = auto()
    MetalTrans = auto()
    Stone = auto()
    Player = auto()
    Trunk = auto()
    TechLaser = auto()
    SlidingWood = auto()
    PlayerOnly = auto()
    Tech = auto()
    TechArmor = auto()
    TechSafe = auto()
    OffZone = auto()
    Bullet = auto()
    TechHook = auto()
    TechGround = auto()
    TechWall = auto()
    TechArrow = auto()
    TechHook2 = auto()
    Forest = auto()
    Wheat = auto()
    TechTarget = auto()
    PavementStair = auto()
    TechTeleport = auto()
    Energy = auto()
    TechMagnetic = auto()
    TurboTechMagnetic_Deprecated = auto()
    Turbo2TechMagnetic_Deprecated = auto()
    TurboWood_Deprecated = auto()
    Turbo2Wood_Deprecated = auto()
    FreeWheelingTechMagnetic_Deprecated = auto()
    FreeWheelingWood_Deprecated = auto()
    TechSuperMagnetic = auto()
    TechNucleus = auto()
    TechMagneticAccel = auto()
    MetalFence = auto()
    TechGravityChange = auto()
    TechGravityReset = auto()
    RubberBand = auto()
    Gravel = auto()
    Hack_NoGrip_Deprecated = auto()
    Bumper2_Deprecated = auto()
    NoSteering_Deprecated = auto()
    NoBrakes_Deprecated = auto()
    RoadIce = auto()
    RoadSynthetic = auto()
    Green = auto()
    Plastic = auto()
    DevDebug = auto()
    Free3 = auto()
    XXX_Null = auto()

class ReactorBoostType(Enum):
    NONE = auto()
    Up = auto()
    Down = auto()
    UpAndDown = auto()

class ReactorBoostLevel(Enum):
    NONE = auto()
    Lvl1 = auto()
    Lvl2 = auto()

class EventType(IntEnum):
    BLOCK_ENTER = 0
    BLOCK_EXIT = 1
    INPUT_CHANGE = 2
    CHECKPOINT = 3
    FINISH = 4

class BlockDirection(IntEnum):
    NONE = auto()
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()