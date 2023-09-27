from sumo_rl.environment.observations import ObservationFunction
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np

class OB1(ObservationFunction):
    
    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        observation = np.array(phase_id + queue + density, dtype=np.float32)
       
        return observation
#
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""

        return spaces.Box(
            low=np.zeros(3 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(3 * len(self.ts.lanes), dtype=np.float32),
        )
    
class OB2(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:

        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
       
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        observation = np.array(phase_id + queue + wait + laneOccupancy, dtype=np.float32)
      
        return observation
    
    def observation_space(self) -> spaces.Box:

        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 3 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 3 * len(self.ts.lanes), dtype=np.float32),
        )

class OB3(ObservationFunction):
    
    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:

        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        observation = np.array(phase_id + queue + wait + laneOccupancy + minDist, dtype=np.float32)

        return observation

    def observation_space(self) -> spaces.Box:

        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 4 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 4 * len(self.ts.lanes), dtype=np.float32),
        )


class OB4(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        # wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        # avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + laneOccupancy, dtype=np.float32)

        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 2 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB5(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        # wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        # avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + laneOccupancy + density, dtype=np.float32)
 
        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 3* len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 3 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB6(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + wait + laneOccupancy + density + avgSpeedPerLane, dtype=np.float32)

        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 5 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 5 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB7(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        # avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + wait + laneOccupancy + queueOut, dtype=np.float32)

        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 4 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 4 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB8(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + wait + laneOccupancy + density + avgSpeedPerLane + minDist, dtype=np.float32)
    
        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 6 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 6 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB9(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + wait + laneOccupancy + density + avgSpeedPerLane + minDist + queueOut, dtype=np.float32)

        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 7 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 7 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB10(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        # wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + laneOccupancy + avgSpeedPerLane, dtype=np.float32)

        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 3 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 3 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB11(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + wait + laneOccupancy + avgSpeedPerLane, dtype=np.float32)
       
        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 4 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 4 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

class OB12(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------

        #Incoming lane data
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # density = self.ts.get_lanes_density() # The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        queue = self.ts.get_lanes_queue() # The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        time_since_last_phase_change = self.ts.get_time_since_last_phase_change()
        wait = self.ts.get_accumulated_waiting_time_per_lane() # Returns the accumulated waiting time per lane.
        laneOccupancy = self.ts.get_occupancy_per_lane() # Returns the occupancy (20 to 35 meters around the intersection) of each lane
        avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + time_since_last_phase_change + queue + wait + laneOccupancy + avgSpeedPerLane, dtype=np.float32)
       
        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 4 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 +  4 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------
