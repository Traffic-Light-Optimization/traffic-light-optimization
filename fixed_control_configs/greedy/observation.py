from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces

class GreedyObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        queue = self.ts.get_lanes_queue()
        observation = np.array(queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes), dtype=np.float32),
            high=np.ones(len(self.ts.lanes), dtype=np.float32),
        )