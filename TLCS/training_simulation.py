import traci
import numpy as np
import random
import timeit

class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, red_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps

        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._red_duration = red_duration

        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs

        self._teleporting_cars = 0
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


    def run(self, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        #self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._max_waiting_time = 0
        old_total_wait = 0
        #old_queue_length = 0
        old_state = -1
        old_action = -1
        directions = ["N", "S", "E", "W"]

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()
            

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            #current_queue_length = self._get_queue_length()

            # Pesos para as diferentes componentes da recompensa
            weight_waiting_time = 1
            #weight_queue_length = 0.2

            # Normalização para garantir que todos os componentes contribuam igualmente
            # max_waiting_time = self._max_steps  # Estimativa do máximo tempo de espera possível
            #max_queue_length = 50    # Estimativa do comprimento máximo da fila possível
            
            # Cálculo das mudanças em cada métrica
            delta_waiting_time = (old_total_wait - current_total_wait) / 10
            #delta_queue_length = (old_queue_length - current_queue_length) / 10

            #delta_waiting_time = 1 if delta_waiting_time > 1 else -1 if delta_waiting_time < -1 else delta_waiting_time
            #delta_queue_length = 1 if delta_queue_length > 1 else -1 if delta_queue_length < -1 else delta_queue_length

            reward = (  delta_waiting_time + (10 if current_total_wait == 0 and not self._teleporting_cars else 0))
            
            #print("Reward:", reward, "- Waiting time:", current_total_wait, "- Delta waiting time:", delta_waiting_time, "- Teleporting cars:", self._teleporting_cars, "- Max waiting time:", self._max_waiting_time)

            if self._teleporting_cars > 0:
                print("Teleporting cars:", self._teleporting_cars)
                print(reward)
                reward = -2 * reward * self._teleporting_cars if reward > 0 else reward * 2 * self._teleporting_cars

                self._teleporting_cars = 0

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if old_action != action:
                self._set_yellow_phase()
                self._simulate(self._yellow_duration)
                self._set_red_phase()
                self._simulate(self._red_duration)

                self._tl_memory_str = self._tl_memory_str.replace(directions[action], "") + directions[action]
                self._time_in_phase = [1, 0, 0, 0]
            elif self._time_in_phase[-1] != 1:
                i = self._time_in_phase.index(1)
                self._time_in_phase[i] = 0
                self._time_in_phase[i + 1] = 1      

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            #old_queue_length = current_queue_length

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        print("Max waiting time:", self._max_waiting_time)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo

            # check if a car has teleported (teleporting cars are those that were spawned in the simulation and were teleported to the end of the route)
            self._teleporting_cars += traci.simulation.getStartingTeleportNumber()

            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["north_in", "south_in", "east_in", "west_in"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)  # compute the waiting time for a single car
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


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


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

        return state


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model._batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store