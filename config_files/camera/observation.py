from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces

class ModelCameraObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # time_since_last_phase_change = self.ts.get_time_since_last_phase_change()
        queues = self.ts.get_lanes_queue_from_detectors()
        occupancy = self.ts.get_lanes_occupancy_from_detectors()
        avg_speeds = self.ts.get_average_lane_speeds_from_detectors()
        wait_times = self.ts.get_accumulated_waiting_time_per_lane_from_detectors()
        # min_dists = self.ts.get_dist_to_intersection_per_lane_from_detectors()
        # pressures = self.ts.get_lanes_pressure_from_detectors()
        observation = np.array(phase_id + queues + occupancy + avg_speeds + wait_times, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 4*len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 4*len(self.ts.lanes), dtype=np.float32),
        )
    
class GreedyCameraObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        queue = self.ts.get_lanes_occupancy_from_detectors()
        observation = np.array(queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes), dtype=np.float32),
            high=np.ones(len(self.ts.lanes), dtype=np.float32),
        )
    
class MaxPressureCameraObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        queue = self.ts.get_lanes_pressure_from_detectors()
        observation = np.array(queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes), dtype=np.float32),
            high=np.ones(len(self.ts.lanes), dtype=np.float32),
        )