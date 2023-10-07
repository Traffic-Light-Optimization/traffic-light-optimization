from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces

class ModelGpsObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        # density = self.ts.get_lanes_density_hidden()
        queue = self.ts.get_lanes_queue_hidden()
        occupancy = self.ts.get_occupancy_per_lane_hidden() #occupancy within 35m
        # wait_times = self.ts.get_accumulated_waiting_time_per_lane_hidden()
        avg_speeds = self.ts.get_average_lane_speeds_hidden()
        # time_since_last_phase_change = self.ts.get_time_since_last_phase_change()

        # print()
        # print(f"Density = {density}")
        # print(f"Queue = {queue}")
        # print(f"Occupancy = {occupancy}")
        # print(f"Wait times = {wait_times}")
        # print(f"Avg speeds = {avg_speeds}")
        # print(f"Phase times = {phase_times}")
        # print()
        observation = np.array(phase_id + queue + occupancy + avg_speeds, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 3*len(self.ts.lanes) + self.ts.num_green_phases, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 3*len(self.ts.lanes) + self.ts.num_green_phases, dtype=np.float32),
        )
    
class GreedyGpsObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        density = self.ts.get_lanes_density_hidden()
        observation = np.array(density, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes), dtype=np.float32),
            high=np.ones(len(self.ts.lanes), dtype=np.float32),
        )
    
class MaxPressureGpsObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        pressure = self.ts.get_lanes_pressure_hidden()
        observation = np.array(pressure, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes), dtype=np.float32),
            high=np.ones(len(self.ts.lanes), dtype=np.float32),
        )