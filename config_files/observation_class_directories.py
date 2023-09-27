from config_files.camera.observation import ModelCameraObservationFunction, GreedyCameraObservationFunction, MaxPressureCameraObservationFunction
from config_files.gps.observation import ModelGpsObservationFunction, GreedyGpsObservationFunction, MaxPressureGpsObservationFunction
from config_files.custom_observation import CustomObservationFunction
from config_files.ob1 import OB1
from config_files.ob2 import OB2
from config_files.ob3 import OB3
from config_files.ob4 import OB4
from config_files.ob5 import OB5
from config_files.ob6 import OB6
from config_files.ob7 import OB7
from config_files.ob8 import OB8
from config_files.ob9 import OB9
from config_files.ob10 import OB10
from config_files.ob11 import OB11

from sumo_rl.environment.observations import DefaultObservationFunction

TYPE_OBSERVATION = {
    "greedy": {
        "camera": GreedyCameraObservationFunction,
        "gps": GreedyGpsObservationFunction,
    },
    "max_pressure": {
        "camera": MaxPressureCameraObservationFunction,
        "gps": MaxPressureGpsObservationFunction,
    },
    "fixed": {
        "camera": DefaultObservationFunction,
        "gps": DefaultObservationFunction
    },
    "rand": {
        "camera": DefaultObservationFunction,
        "gps": DefaultObservationFunction,
    },
    "model": {
        "camera": ModelCameraObservationFunction,
        "gps": ModelGpsObservationFunction,
        "custom": CustomObservationFunction,
        "ob1": OB1,
        "ob2": OB2,
        "ob3": OB3,
        "ob4": OB4,
        "ob5": OB5,
        "ob6": OB6,
        "ob7": OB7,
        "ob8": OB8,
        "ob9": OB9,
        "ob10": OB10,
        "ob11": OB11,

    }
}

def get_observation_class(type: str, observation: str):
    return TYPE_OBSERVATION[type][observation]