import traci
import numpy as np
import timeit

class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, red_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._red_duration = red_duration

        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

        self._tl_memory_str = "NSEW"
        self._memory_code = {
                                'NESW': [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                'NEWS': [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                                'NSEW': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                'NSWE': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                                'NWES': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                                'NWSE': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                                'ENSW': [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                'ENWS': [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                                'ESNW': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                'ESWN': [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                'EWNS': [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                                'EWSN': [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                'SNEW': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                'SNWE': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                                'SENW': [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                'SEWN': [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                'SWNE': [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                                'SWEN': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                                'WNES': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                                'WNSE': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                'WENS': [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                                'WESN': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                'WSNE': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                'WSEN': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
                            }
        self._time_in_phase = [1, 0, 0, 0]

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._max_waiting_time = 0
        old_total_wait = 0
        old_action = -1 # dummy init
        directions = ["N", "S", "E", "W"]

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)
            print("Action:", directions[action])

            # if the chosen phase is different from the last phase, activate the yellow phase
            if old_action != action:
                self._set_yellow_phase()
                self._simulate(self._yellow_duration)
                self._set_red_phase()
                self._simulate(self._red_duration)

                self._tl_memory_str = self._tl_memory_str.replace(directions[action], "") + directions[action]
                print("Memory String:", self._tl_memory_str)
                self._time_in_phase = [1, 0, 0, 0]
            elif self._time_in_phase[-1] != 1:
                i = self._time_in_phase.index(1)
                print("i =", i)
                self._time_in_phase[i] = 0
                self._time_in_phase[i+1] = 1


            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        print("Total reward:", np.sum(self._reward_episode), "Max waiting time:", self._max_waiting_time)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["north_in", "south_in", "east_in", "west_in"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
                if wait_time > self._max_waiting_time:
                    self._max_waiting_time = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self):
        """
        Activate the yellow phase of the old_action
        """
        yellow_phase = traci.trafficlight.getRedYellowGreenState('TL').replace('G', 'y')
        traci.trafficlight.setRedYellowGreenState('TL', yellow_phase)
    

    def _set_red_phase(self):
        """
        Activate the red phase of the old_action
        """
        traci.trafficlight.setRedYellowGreenState('TL', 'rrrrrrrrrrrrrrrr')


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        phases = ['GGGGrrrrrrrrrrrr', 'rrrrrrrrGGGGrrrr', 'rrrrGGGGrrrrrrrr', 'rrrrrrrrrrrrGGGG']
        traci.trafficlight.setRedYellowGreenState('TL', phases[action_number])


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        incoming_lanes = ["north_in", "south_in", "east_in", "west_in"]
        queue_length = sum([traci.edge.getLastStepHaltingNumber(lane) for lane in incoming_lanes])
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(40)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = 100 - traci.vehicle.getLanePosition(car_id) # distance between car and intersection
            lane_id = traci.vehicle.getLaneID(car_id)

            lane_cell = lane_pos // 10 # 10 meters is the length of a cell
            lane_cell = 9 if lane_cell > 9 else lane_cell
        
            incoming_lanes = ['north_in_0', 'south_in_0', 'east_in_0', 'west_in_0']
            lane_group = incoming_lanes.index(lane_id) if lane_id in incoming_lanes else None

            if lane_group is not None:
                car_position = int(lane_group*10 + lane_cell) # composition of the two position ID to create a number in inteval [0, 39]
                valid_car = True
            else:
                valid_car = False   # Flag for not considering the cars croossing the intersection or driving away from it
            
            if valid_car:
                state[car_position] = 1  # Write the position of the car car_id in the array in the form of "cell occupied"
        
        state = np.concatenate((state, self._memory_code[self._tl_memory_str], self._time_in_phase)) # Add the traffic light phase to the state array

        print("North:", state[:10])
        print("South:", state[10:20])
        print("East:", state[20:30])
        print("West:", state[30:40])

        print("N:", state[40:44])
        print("S:", state[44:48])
        print("E:", state[48:52])
        print("W:", state[52:56])

        print("Time in phase:", state[56:60])

        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



