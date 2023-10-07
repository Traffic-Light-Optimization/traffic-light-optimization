from sumo_rl.environment.env import SumoEnvironment
from config_files.greedy.action import greedy_action
from config_files.max_pressure.action import max_pressure_action
from config_files.action_lane_relationships import get_action_lane_relationships
from config_files.net_route_directories import get_file_locations
from config_files.observation_class_directories import get_observation_class
from config_files import reward_directories
import csv

types = ["fixed", "greedy", "max_pressure", "rand"]
maps = ["cologne1", "cologne3", "cologne8"] #choose the map to simulate
gui = False #SUMO gui
reward_option = 'defandspeed'  # 'default', 'defandmaxgreen','speed','defandspeed','defandpress','all3','avgwait','avgwaitavgspeed','defandaccumlatedspeed', 'defandmaxgreen'
num_seconds = 3600 #episode duration
delta_time = 8 #step duration
max_green = 60
yellow_time = 3 # min yellow time
seed = "12345"

for map_name in maps:
    map = get_file_locations(map_name) #obtain network, route, and additional files
    action_lanes = get_action_lane_relationships(map_name) #dict of relationships between actions and lanes for each intersection
    
    for type in types:
        observations = ["camera", "gps", "ideal"] if type != "fixed" and type != "rand" else ["none"]

        for observation in observations:
            
            print(f"Start: {map_name}-{type}-{observation}")
            
            # Selects the observation class specified
            observation_class = get_observation_class(type, observation)

            # Get the corresponding reward function based on the option
            reward_function = reward_directories.reward_functions.get(reward_option)

            env = SumoEnvironment(
                net_file=map["net"],
                route_file=map["route"],
                use_gui=gui,
                num_seconds=num_seconds,
                delta_time=delta_time,
                max_green=max_green,
                sumo_seed=seed,
                add_per_agent_info = True,
                observation_class=observation_class,
                reward_fn=reward_function,
                yellow_time = yellow_time,
                additional_sumo_cmd=f"--additional-files {map['additional']}" if observation == "camera" else None,
                fixed_ts = True if type == "fixed" else False,
                hide_cars = True if observation == "gps" else False
            )

            data = [] #initialize a list to store the data
            observations = env.reset()
            done = False
            avg_rewards = []
            while not done:
                if type == "greedy":
                    actions = {agent: greedy_action(observations[agent], action_lanes[agent]) for agent in env.ts_ids}
                elif type == "max_pressure":
                    actions = {agent: max_pressure_action(observations[agent], action_lanes[agent]) for agent in env.ts_ids}
                elif type == "fixed":
                    actions = {}
                elif type == "rand":
                    actions = {agent: env.action_spaces(agent).sample() for agent in env.ts_ids}
                else:
                    raise ValueError(f"{type} is an invalid type for fixed control simulations")
                observations, rewards, dones, infos = env.step(actions)
                if type != "fixed":
                    avg_rewards.append(sum(rewards.values())/len(rewards.values()))
                data.append(infos.copy())
                done = dones['__all__']

            if type != "fixed":
                mean_reward = sum(avg_rewards)/len(avg_rewards)
                print(f"Mean reward for simulation = {mean_reward}")

            env.close()

            # Create a CSV file and write the data to it
            headings = data[0].keys()
            if data:
                with open(f"./results/{type}/{map_name}-{type}-{observation}_conn1.csv", mode='w', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=headings)
                    writer.writeheader()
                    writer.writerows(data)