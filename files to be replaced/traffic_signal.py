"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_pressure = 0.0
        self.last_avg_speed = 0.0
        self.last_accumulated_speed = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}
        all_laneareas = self.sumo.lanearea.getIDList()
        self.laneareas = []
        for lane in self.lanes:
            for lanearea in all_laneareas:
                if self.sumo.lanearea.getLaneID(lanearea) == lane:
                    self.laneareas.append(lanearea)
        self.prev_lane_vehicle_ids = {lane: [] for lane in self.lanes} #dict of vehicle ids in each lane
        self.prev_lanearea_vehicle_ids = {lanearea: [] for lanearea in self.laneareas} #dict of vehicle ids in each lanearea

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)
        self.time_since_phase_selected = [0 for _ in range(self.num_green_phases)] ###TESTING

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        for i in range(len(self.time_since_phase_selected)):
            self.time_since_phase_selected[i] += 1 
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else: #changes to a new phase
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.time_since_phase_selected[new_phase] = 0 ###TESTING
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward
    
    def diff_pressure_reward(self):
        """Compute the difference in pressure between the current and the previous time step."""
        current_pressure = self.get_pressure()
        diff = current_pressure - self.last_pressure if hasattr(self, 'last_pressure') else 0.0
        self.last_pressure = current_pressure
        return diff

    def diff_speed_reward(self):
        current_accumulated_speed = self.get_accumulated_speed()
        diff_speed = current_accumulated_speed - self.last_accumulated_speed
        self.last_accumulated_speed = current_accumulated_speed
        return diff_speed

    def diff_avg_speed_reward(self):
        """Compute the difference in average speed between the current and the previous time step."""
        current_avg_speed = self.get_average_speed()
        diff = current_avg_speed - self.last_avg_speed if hasattr(self, 'last_avg_speed') else 0.0
        self.last_avg_speed = current_avg_speed
        return diff
    
    def time_since_phase_chosen_reward(self): ###TESTING
        """Returns a punishment if certain phases have not been selected for a long time"""
        return -sum(self.time_since_phase_selected) / (self.num_green_phases * self.max_green)
    
    def max_green_reward(self): ###TESTING
        """Returns a punishment if the agent hasn't changed its phase in max_green seconds"""
        if self.time_since_last_phase_change > self.max_green and self.max_green > 0:
            return -(self.time_since_last_phase_change - self.max_green) / self.max_green
        else:
            return 0
    
    def reward_highest_occupancy_phase(self):
        """Rewards a prediction that chooses a green phase for the lane with the highest occupancy if possible."""
        lane_occupancy = self.get_occupancy_per_lane()
        # Check if lane_occupancy is a vector of zeros
        if all(occupancy == 0 for occupancy in lane_occupancy):
            return 0.0  # If all lanes have zero occupancy, return a reward of 0
        # Find the index of the lane with the highest occupancy
        highest_occupancy_lane = np.argmax(lane_occupancy)
        # Check if the current green phase corresponds to the lane with the highest occupancy
        if highest_occupancy_lane == self.green_phase:
            return 0.1  # If the current green phase is already the highest occupancy lane, reward 1
        return -0.1  # Otherwise, reward 0

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation
    
    ###TESTING
    def get_time_since_last_phase_change(self):
        return [self.time_since_last_phase_change/self.max_green]
    
    def get_dist_to_intersection_per_lane(self):
        min_dist = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            lane_length = self.lanes_length[lane]
            if veh_list:
                distances = [lane_length - self.sumo.vehicle.getLanePosition(veh) for veh in veh_list]
                min_distance = round(min(distances),5)
                min_dist.append(min_distance/lane_length)
            else:
                min_dist.append(1)  # No vehicles in the lane, set distance to max
        return min_dist
    
    ###TESTING
    def get_dist_to_intersection_per_lane_from_detectors(self):
        min_dist = []
        for lanearea in self.laneareas:
            veh_list = self.sumo.lanearea.getLastStepVehicleIDs(lanearea)
            lane_length = self.sumo.lane.getLength(self.sumo.lanearea.getLaneID(lanearea))
            lanearea_length = self.sumo.lanearea.getLength(lanearea)
            if veh_list:
                distances = [lane_length - self.sumo.vehicle.getLanePosition(veh) for veh in veh_list]
                min_distance = round(min(distances),5)
                min_dist.append(min_distance/lanearea_length)
            else:
                min_dist.append(1)  # No vehicles in the lane, set distance to max
        return min_dist
    
    def get_times_since_phase_selected(self) -> List[int]:
        times = []
        for time in self.time_since_phase_selected:
            times.append(time / (self.num_green_phases * self.max_green))
        return times


    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(round(wait_time,5))
        return wait_time_per_lane
    
    def get_accumulated_waiting_time_list(self) -> List[float]:
        """Returns the accumulated waiting time of each car
        """
        wait_time_list = []
        
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = []
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time.append(round(self.env.vehicles[veh][veh_lane],5))
            wait_time_list = wait_time_list + wait_time
        return wait_time_list
    
    ###TESTING
    def get_accumulated_waiting_time_per_lane_from_detectors(self) -> List[float]:
        wait_time_per_lane = []
        for lanearea in self.laneareas:
            veh_list = self.sumo.lanearea.getLastStepVehicleIDs(lanearea)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(round(wait_time,5))
        return wait_time_per_lane
    
    ###TESTING
    def get_accumulated_waiting_time_per_lane_hidden(self) -> List[float]:
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            visible_veh_list = []
            for veh_id in veh_list:
                if self.sumo.vehicle.getColor(veh_id) != (255, 255, 255, 255):
                    visible_veh_list.append(veh_id)
            wait_time = 0.0
            for veh in visible_veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(round(wait_time,5))
        return wait_time_per_lane
    
    def get_occupancy_per_lane(self) -> List[float]:
        """Calculate and return the occupancy of the nearest 35m or less section of each lane.

        Occupancy is defined as the number of cars in 35m of the lane closest to the intersection
        divided by the number of cars that could fit in that 35m.

        Returns:
            List[float]: List of occupancy values for each lane.
        """
        max_length = 35
        lane_occupancy = []
        for lane in self.lanes:
            
            lane_length = self.lanes_length[lane]
            lane_area_length = lane_length if lane_length < max_length else max_length
                
            # Get the list of vehicle IDs in the lane
            vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane)

            # Calculate the number of vehicles in the specified section of the lane
            num_vehicles_in_section = 0
            for veh_id in vehicle_ids:
                if lane_length - self.sumo.vehicle.getLanePosition(veh_id) <= lane_area_length:
                    num_vehicles_in_section += 1

            # Calculate the number of vehicles that could fit in the section
            max_vehicles_in_section = np.ceil(lane_area_length / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))

            # Calculate the occupancy (number of vehicles in the section / maximum vehicles in the section)
            occupancy = num_vehicles_in_section / max_vehicles_in_section if max_vehicles_in_section > 0 else 0.0

            lane_occupancy.append(round(occupancy, 5))

        return lane_occupancy

    ###TESTING
    def get_occupancy_per_lane_hidden(self) -> List[float]:
        max_length = 35
        lane_occupancy = []
        for lane in self.lanes:
            
            lane_length = self.lanes_length[lane]
            lane_area_length = lane_length if lane_length < max_length else max_length
                
            # Get the list of vehicle IDs in the lane
            vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane)
            visible_vehicle_ids = []
            for veh_id in vehicle_ids:
                if self.sumo.vehicle.getColor(veh_id) != (255, 255, 255, 255):
                    visible_vehicle_ids.append(veh_id)

            # Calculate the number of vehicles in the specified section of the lane
            num_vehicles_in_section = 0
            for veh_id in visible_vehicle_ids:
                if lane_length - self.sumo.vehicle.getLanePosition(veh_id) <= lane_area_length:
                    num_vehicles_in_section += 1

            # Calculate the number of vehicles that could fit in the section
            max_vehicles_in_section = np.ceil((lane_area_length + self.MIN_GAP) / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))

            # Calculate the occupancy (number of vehicles in the section / maximum vehicles in the section)
            occupancy = num_vehicles_in_section / max_vehicles_in_section if max_vehicles_in_section > 0 else 0.0

            lane_occupancy.append(round(occupancy, 5))

        return lane_occupancy

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed +=  np.sqrt(self.sumo.vehicle.getSpeed(v)**2 + self.sumo.vehicle.getLateralSpeed(v)**2) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)
    
    def get_accumulated_speed(self) -> float:
        """Returns the accumulated speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        acc_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            acc_speed +=  np.sqrt(self.sumo.vehicle.getSpeed(v)**2 + self.sumo.vehicle.getLateralSpeed(v)**2) / self.sumo.vehicle.getAllowedSpeed(v)
        return acc_speed
    
    def get_average_speed_list(self) -> List[float]:
        avg_speed = []
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return []
        else:
            avg_speed =  [np.sqrt(self.sumo.vehicle.getSpeed(v)**2 + self.sumo.vehicle.getLateralSpeed(v)**2) for v in vehs]
        return avg_speed

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection divided by the total number of vehicles."""
        # Calculate the total number of vehicles leaving the intersection
        vehicles_leaving = sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)
        
        # Calculate the total number of vehicles approaching the intersection
        vehicles_approaching = sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        
        # Calculate the pressure divided by the total number of vehicles
        if vehicles_approaching > 0:
            pressure_normalized = (vehicles_leaving - vehicles_approaching) / (vehicles_approaching + vehicles_leaving)
        else:
            pressure_normalized = 0.0  # Avoid division by zero
        
        return pressure_normalized

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    ###TESTING
    def get_lanes_occupancy_from_detectors(self) -> List[List[str]]:
        occupancies = []
        for lanearea in self.laneareas:
            vehicle_ids = self.sumo.lanearea.getLastStepVehicleIDs(lanearea)
            lanearea_length = self.sumo.lanearea.getLength(lanearea)
            max_vehicles = np.ceil((lanearea_length + self.MIN_GAP) / (self.MIN_GAP + self.sumo.lane.getLastStepLength(self.sumo.lanearea.getLaneID(lanearea))))
            occupancies.append(len(vehicle_ids) / max_vehicles)
        return occupancies
    
    ###TESTING
    def get_lanes_pressure_from_detectors(self) -> List[str]:
        current_vehicle_ids = {lane_area: self.sumo.lanearea.getLastStepVehicleIDs(lane_area) for lane_area in self.laneareas}
        pressures = []
        for lanearea, vehicle_ids in self.prev_lanearea_vehicle_ids.items():
            outgoing_cars = 0
            for vehicle_id in vehicle_ids:
                if vehicle_id not in current_vehicle_ids[lanearea]:
                    outgoing_cars += 1
            incoming_cars = len(current_vehicle_ids[lanearea])
            pressure = incoming_cars - outgoing_cars
            pressures.append(pressure)
        self.prev_lanearea_vehicle_ids = current_vehicle_ids
        return pressures
        
    def get_average_lane_speeds(self) -> List[float]:
        """Returns a list of the average speed of vehicles in each lane normalized by the maximum allowed speed.

        Returns:
            List[float]: List of average lane speeds for each incoming lane.
        """
        average_speeds = []

        for lane in self.lanes:
            vehicles_in_lane = self.sumo.lane.getLastStepVehicleIDs(lane)

            if vehicles_in_lane:
                total_speed = 0.0
                total_allowed_speed = 0.0

                for vehicle_id in vehicles_in_lane:
                    vehicle_speed = np.sqrt(self.sumo.vehicle.getSpeed(vehicle_id)**2 + self.sumo.vehicle.getLateralSpeed(vehicle_id)**2)
                    vehicle_allowed_speed = self.sumo.vehicle.getAllowedSpeed(vehicle_id)

                    total_speed += vehicle_speed
                    total_allowed_speed += vehicle_allowed_speed

                # Calculate the average speed for the lane and normalize by the maximum allowed speed
                if total_allowed_speed > 0:
                    lane_average_speed = total_speed / total_allowed_speed
                else:
                    lane_average_speed = 0.0

                average_speeds.append(lane_average_speed)
            else:
                # If no vehicles in the lane, set the average speed to 0
                average_speeds.append(1.0)

        return average_speeds
    
    ###TESTING
    def get_average_lane_speeds_hidden(self) -> List[float]:
        average_speeds = []

        for lane in self.lanes:
            vehicles_in_lane = self.sumo.lane.getLastStepVehicleIDs(lane)
            visible_vehicles_in_lane = []
            for veh_id in vehicles_in_lane:
                if self.sumo.vehicle.getColor(veh_id) != (255, 255, 255, 255):
                    visible_vehicles_in_lane.append(veh_id)

            if visible_vehicles_in_lane:
                total_speed = 0.0
                total_allowed_speed = 0.0

                for vehicle_id in visible_vehicles_in_lane:
                    vehicle_speed = np.sqrt(self.sumo.vehicle.getSpeed(vehicle_id)**2 + self.sumo.vehicle.getLateralSpeed(vehicle_id)**2)
                    vehicle_allowed_speed = self.sumo.vehicle.getAllowedSpeed(vehicle_id)

                    total_speed += vehicle_speed
                    total_allowed_speed += vehicle_allowed_speed

                # Calculate the average speed for the lane and normalize by the maximum allowed speed
                if total_allowed_speed > 0:
                    lane_average_speed = total_speed / total_allowed_speed
                else:
                    lane_average_speed = 0.0

                average_speeds.append(lane_average_speed)
            else:
                # If no vehicles in the lane, set the average speed to 0
                average_speeds.append(1.0)

        return average_speeds
    
    ###TESTING
    def get_average_lane_speeds_from_detectors(self) -> List[float]:
        average_speeds = []

        for lanearea in self.laneareas:
            vehicles_in_lane = self.sumo.lanearea.getLastStepVehicleIDs(lanearea)

            if vehicles_in_lane:
                total_speed = 0.0
                total_allowed_speed = 0.0

                for vehicle_id in vehicles_in_lane:
                    vehicle_speed = np.sqrt(self.sumo.vehicle.getSpeed(vehicle_id)**2 + self.sumo.vehicle.getLateralSpeed(vehicle_id)**2)
                    vehicle_allowed_speed = self.sumo.vehicle.getAllowedSpeed(vehicle_id)

                    total_speed += vehicle_speed
                    total_allowed_speed += vehicle_allowed_speed

                # Calculate the average speed for the lane and normalize by the maximum allowed speed
                if total_allowed_speed > 0:
                    lane_average_speed = total_speed / total_allowed_speed
                else:
                    lane_average_speed = 0.0

                average_speeds.append(lane_average_speed)
            else:
                # If no vehicles in the lane, set the average speed to 0
                average_speeds.append(1.0)

        return average_speeds

    ###TESTING
    def get_lanes_pressure_hidden(self) -> List[str]:
        pressures = []
        lanes_vehicle_ids = {lane: self.sumo.lane.getLastStepVehicleIDs(lane) for lane in self.lanes} #Dict of all vehicle ids
        current_vehicle_ids = {lane: [] for lane in self.lanes} #Dict of visible vehicle ids
        for lane, vehicle_ids in lanes_vehicle_ids.items():
            for id in vehicle_ids:
                if self.sumo.vehicle.getColor(id) != (255, 255, 255, 255):
                    current_vehicle_ids[lane].append(id)
        for lane, vehicle_ids in self.prev_lane_vehicle_ids.items():
            outgoing_cars = 0
            for id in vehicle_ids:
                if id not in current_vehicle_ids[lane]:
                    outgoing_cars += 1
            incoming_cars = len(current_vehicle_ids[lane])
            pressure = incoming_cars - outgoing_cars
            pressures.append(pressure)
        self.prev_lane_vehicle_ids = current_vehicle_ids
        return pressures


    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [round(min(1, density), 5) for density in lanes_density]
    
    ###TESTING
    def get_lanes_density_hidden(self) -> List[float]:
        lanes_vehicle_ids = {lane: list(self.sumo.lane.getLastStepVehicleIDs(lane)) for lane in self.lanes}
        results_lanes_vehicle_ids = {lane: [] for lane in self.lanes}
        for lane, lane_vehicle_ids in lanes_vehicle_ids.items():
            for vehicle_id in lane_vehicle_ids:
                if self.sumo.vehicle.getColor(vehicle_id) != (255, 255, 255, 255):
                    results_lanes_vehicle_ids[lane].append(vehicle_id)
        lane_densities = [len(results_lanes_vehicle_ids[lane]) / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in self.lanes]
        return lane_densities

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]
    
    def get_outgoing_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the outgoing lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, queue) for queue in lanes_queue]
    
    ###TESTING
    def get_lanes_queue_hidden(self) -> List[float]:
        lanes_vehicle_ids = {lane: list(self.sumo.lane.getLastStepVehicleIDs(lane)) for lane in self.lanes}
        results_lanes_vehicle_ids = {lane: [] for lane in self.lanes}
        for lane, lane_vehicle_ids in lanes_vehicle_ids.items():
            for vehicle_id in lane_vehicle_ids: 
                speed = np.sqrt((self.sumo.vehicle.getLateralSpeed(vehicle_id))**2 + (self.sumo.vehicle.getSpeed(vehicle_id))**2)
                if((self.sumo.vehicle.getColor(vehicle_id) != (255, 255, 255, 255)) and (speed < 0.1)):
                    results_lanes_vehicle_ids[lane].append(vehicle_id)
        lane_queues = [len(results_lanes_vehicle_ids[lane]) / np.ceil((self.lanes_length[lane] + self.MIN_GAP) / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in self.lanes]
        return lane_queues
    
    ###TESTING
    def get_lanes_queue_from_detectors(self) -> List[float]:
        lanes_vehicles_ids = {lanearea: list(self.sumo.lanearea.getLastStepVehicleIDs(lanearea)) for lanearea in self.laneareas}
        results_lanes_vehicle_ids = {lanearea: [] for lanearea in self.laneareas}
        for lanearea, lane_vehicle_ids in lanes_vehicles_ids.items():
            for vehicle_id in lane_vehicle_ids:
                speed = np.sqrt((self.sumo.vehicle.getLateralSpeed(vehicle_id))**2 + (self.sumo.vehicle.getSpeed(vehicle_id))**2)
                if speed < 0.1:
                    results_lanes_vehicle_ids[lanearea].append(vehicle_id)
        lane_queues = [len(results_lanes_vehicle_ids[lanearea]) / np.ceil((self.sumo.lanearea.getLength(lanearea) + self.MIN_GAP) / (self.MIN_GAP + self.sumo.lane.getLastStepLength(self.sumo.lanearea.getLaneID(lanearea)))) for lanearea in self.laneareas]
        return lane_queues
  
    def get_cars_present(self) -> List[str]:
        """Returns the number of cars present in each lane."""
        return [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list
    
    def get_id(self) -> str:
        return self.id

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }