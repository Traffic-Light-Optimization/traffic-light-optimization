from sumo_rl.environment.observations import ObservationFunction
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np

class OB4(ObservationFunction):
    """Custom observation function for traffic signals."""

    ## Default observation 

        # The default observation for each traffic signal agent is a vector:
        # obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

            # phase_one_hot is a one-hot encoded vector indicating the current active green phase
            # min_green is a binary variable indicating whether min_green seconds have already passed in the current phase
            # lane_i_density is the number of vehicles in incoming lane i dividided by the total capacity of the lane
            # lane_i_queue is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

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
        # avgSpeedPerLane = self.ts.get_average_lane_speeds() # returns the average speed of the vehicles in each lane
        # minDist = self.ts.get_dist_to_intersection_per_lane() # returns the distance of the closest car to the intersection for each lane

        #Outgoing lane data
        # queueOut = self.ts.get_outgoing_lanes_queue() #returns the number of vehicles halting divided by the total number of vehicles that can fit in the outgoing lanes. This prevents the model from prioritizing phases when the cars are unable to flow through the intersection into the outgoing lanes.
        observation = np.array(phase_id + queue + laneOccupancy, dtype=np.float32)
        # observation = np.array(phase_id + min_green + density + queue + queueOut + wait + minDist + laneOccupancy + avgSpeedPerLane, dtype=np.float32)
        # print(f"Custom observation {self.ts.get_id()}")
        # print("=========================")
        # print(avgSpeedPerLane)
        # print(min_green)
        # print(queueOut)
        # print(laneOccupancy)
        # print(avgSpeed) 
        # print(phase_id)
        # print(min_green)
        # print(density)
        # print(queue)
        # print(wait)
        # print(minDist)
        # print(observation)
        # print("end")
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


       