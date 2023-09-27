from config_files.camera.observation import ModelCameraObservationFunction, GreedyCameraObservationFunction, MaxPressureCameraObservationFunction
from config_files.gps.observation import ModelGpsObservationFunction, GreedyGpsObservationFunction, MaxPressureGpsObservationFunction
from config_files.custom_observation import CustomObservationFunction
from config_files.idealistic.observation import ModelIdealObservationFunction, GreedyIdealObservationFunction, MaxPressureIdealObservationFunction
from config_files.test.observation import OB1, OB2, OB3, OB4, OB5, OB6, OB7, OB8, OB9, OB10, OB11, OB12

from sumo_rl.environment.observations import DefaultObservationFunction

TYPE_OBSERVATION = {
    "greedy": {
        "camera": GreedyCameraObservationFunction,
        "gps": GreedyGpsObservationFunction,
        "ideal": GreedyIdealObservationFunction,
    },
    "max_pressure": {
        "camera": MaxPressureCameraObservationFunction,
        "gps": MaxPressureGpsObservationFunction,
        "ideal": MaxPressureIdealObservationFunction,
    },
    "fixed": {
        "camera": DefaultObservationFunction,
        "gps": DefaultObservationFunction,
    },
    "rand": {
        "camera": DefaultObservationFunction,
        "gps": DefaultObservationFunction,
    },
    "model": {
        "camera": ModelCameraObservationFunction,
        "gps": ModelGpsObservationFunction,
        "custom": CustomObservationFunction,
        "ideal": ModelIdealObservationFunction,
        # Test observations that will be removed later
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
        "ob12": OB12,
    },
     "test": {
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
        "ob12": OB12,
    }
}

def get_observation_class(type: str, observation: str):
    return TYPE_OBSERVATION[type][observation]