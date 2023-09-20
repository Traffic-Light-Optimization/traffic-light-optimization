from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces

class GpsObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        density = self.ts.get_lanes_density_hidden()
        queue = self.ts.get_lanes_queue_hidden()
        observation = np.array(density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(2*len(self.ts.lanes), dtype=np.float32),
            high=np.ones(2*len(self.ts.lanes), dtype=np.float32),
        )