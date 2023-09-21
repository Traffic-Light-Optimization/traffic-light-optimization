from config_files.camera.observation import ModelCameraObservationFunction, GreedyCameraObservationFunction, MaxPressureCameraObservationFunction
from config_files.gps.observation import ModelGpsObservationFunction, GreedyGpsObservationFunction, MaxPressureGpsObservationFunction
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
        "gps": ModelGpsObservationFunction
    }
}

def get_observation_class(type: str, observation: str):
    return TYPE_OBSERVATION[type][observation]