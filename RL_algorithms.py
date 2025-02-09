import statistics

import numpy as np
import random
import time
import statistics
import pandas as pd

class DeterministicEnvTools:
    def __init__(self, states_list, actions_dict, rewards_dict, transitions_dict, initial_state):
        self.all_states = states_list         # All possible states (that were observed) in a list
        self.all_actions = actions_dict       # Dict in the form of: {state: [actions]}
        self.all_rewards = rewards_dict       # Dict in the form of: {(state, action): reward}
        self.transitions = transitions_dict   # Dict in the form of: {(state, action): nextstate}
        self.initial_state = initial_state
        self.state = initial_state

    def observe_next_state(self, action):
        if action not in self.all_actions[self.state]:
            ret_str = "Can't choose this action in the current state of the environment"
            print("Env says: Can't choose this action in the current state of the environment")
            return ret_str
        else:
            next_state = self.transitions[(self.state, action)]
            return next_state

    def step(self, action):
        self.state = self.observe_next_state(action)
        return None

    def construct_env_from_observations_dict(self, observations_dict, arcs_for_state=5):
        """
        Construct the environment
        :param observations_dict: dict in the format of the output of the function
                                  "MIDI_coding.format_dataset_single_note_optional_modulo_encoding"
        :return: None
        """
        timer_start = time.time()
        t_rewards_dict = {}
        t_actions_dict = {}
        t_transitions_dict = {}
        t_all_states = []
        t_all_states_set = set(t_all_states)
        for outer_key in observations_dict.keys():
            if outer_key not in t_all_states_set:
                t_all_states.append(outer_key)
                t_all_states_set.add(outer_key)
            occurrences = list(observations_dict[outer_key].values())
            occurrences.sort(reverse=True)
            highest_values = occurrences[:arcs_for_state:]
            highest_keys = []
            sum_highest_keys = 0
            t_actions_dict[outer_key] = []
            for inner_key in observations_dict[outer_key]:
                if observations_dict[outer_key][inner_key] in highest_values:
                    t_actions_dict[outer_key].append(inner_key)
                    if inner_key not in t_all_states_set:
                        t_all_states.append(inner_key)
                        t_all_states_set.add(inner_key)
                    highest_keys.append(inner_key)
                    sum_highest_keys += observations_dict[outer_key][inner_key]
            for inner_key in highest_keys:
                t_rewards_dict[(outer_key, inner_key)] = observations_dict[outer_key][inner_key] / sum_highest_keys
                t_transitions_dict[(outer_key, inner_key)] = inner_key
        self.all_states = t_all_states
        self.all_actions = t_actions_dict
        self.all_rewards = t_rewards_dict
        self.transitions = t_transitions_dict
        if self.initial_state == -1:
            i = random.randint(0, len(self.all_states) - 1)
            self.state = self.all_states[i]
        else:
            self.state = self.initial_state
        timer_end = time.time()
        calc_time = timer_end - timer_start

        vec_int_map = {}
        int_vec_map = {}

        curr_int = 0
        for item in self.all_states:
            vec_int_map[item] = curr_int
            int_vec_map[curr_int] = item
            curr_int += 1

        self.vec_int_map = vec_int_map
        self.int_vec_map = int_vec_map

        self.actions_dict_new = {}

        for key in self.all_actions.keys():
            self.actions_dict_new[vec_int_map[key]] = []
            for item in self.all_actions[key]:
                self.actions_dict_new[vec_int_map[key]].append(vec_int_map[item])

        self.rewards_dict_new = {}

        for key in self.all_rewards.keys():
            new_key = (vec_int_map[key[0]], vec_int_map[key[1]])
            self.rewards_dict_new[new_key] = self.all_rewards[key]

        print("EnvironmentTools constructed in " + str(round(calc_time, 2)) + " seconds")
        return None



class AgentTools:
    def __init__(self, env, horizon, policy_dict={}, value_func={}, q_func={}, best_q={}, state_classes={}, time=0, discount_factor=1, alpha=0.9, lambd=0.9):
        self.policy = policy_dict                    # Dict in the form of: {state: action}
        self.env = env                               # DeterministicEnv object
        self.V = value_func                          # Dict in the form of: {state: E[return]}
        self.Q = q_func                              # Dict in the form of: {(state, action): E[return]}
        self.greedy_Q_val = best_q                   # Dict in the form of: {state: [best action, it's Q]}
        self.state_classes = state_classes           # Dict in the form of: {class: list of states}
        self.log = {}
        self.horizon = horizon
        self.t = time
        self.discount_factor = discount_factor
        self.lambd = lambd
        self.alpha = alpha
        self.times_without_improvement = 0

        self.one_tuple_int_map = {}
        self.int_one_tuple_map = {}
        self.two_tuple_int_map = {}
        self.int_two_tuple_map = {}

        self.Q_vec = np.array([0])                   # Q as a np array (vector)
        self.Q_mat = np.array([[0, 0], [0, 0]])      # Q as a np array (matrix)
        self.Q_state_vec = np.array([0])            # np array - for each encoded[(state, action)] -> encoded[state]
        self.Q_act_vec = np.array([0])              # np array - for each encoded[(state, action)] -> encoded[action]
        self.e_vec = np.array([0])                   # Eligibility traces - np array (vector)
        self.e_mat = np.array([[0, 0], [0, 0]])      # Eligibility traces - np array (matrix)

    def standard_state_classification(self, state, new_avg_mode=False):
        ret_lst = []
        last_item = -1
        for idx, item in enumerate(state):
            if idx == 0:
                ret_lst.append(999)   # class must start at 999
            else:
                if item != 666:
                    if item > last_item:
                        ret_lst.append(1)
                    elif item < last_item:
                        ret_lst.append(-1)
                    else:   # item == last_item
                        ret_lst.append(0)
                else:     # item == 666   (no note is played)
                    ret_lst.append(666)
            if item != 666:  # A note is played
                last_item = item
            else:  # No note is played - choose pseudo last_item
                if new_avg_mode:
                    state_no_666 = [i for i in state if i < 128]
                    if len(state_no_666) == 0:
                        state_no_666 = [72]    # If C5 not good for the scale - change
                    last_item = int(statistics.mean(state_no_666))
                else:
                    last_item = int(statistics.mean(state))
        return tuple(ret_lst)

    def construct_agent_from_env(self, map_tuples_to_int=1, random_V_Q=1):
        timer_start = time.time()
        t_policy = {}
        if map_tuples_to_int == 1:
            temp_lst = [0 for item in self.env.all_rewards]
            t_Q_vec = np.array(temp_lst, dtype=np.float64)
            self.e_vec = np.array(temp_lst, dtype=np.float64)
            self.e_mat = np.zeros(shape=(len(self.env.all_actions.keys()), len(self.env.all_actions.keys())))
        curr_int = 0
        for key in self.env.all_actions.keys():
            if map_tuples_to_int == 1:
                self.one_tuple_int_map[key] = curr_int
                self.int_one_tuple_map[curr_int] = key
                curr_int += 1
            if random_V_Q == 1:
                self.V[key] = random.uniform(0, 1)
            else:
                self.V[key] = 0
            curr_class = self.standard_state_classification(key)
            if curr_class not in self.state_classes.keys():
                self.state_classes[curr_class] = []
            self.state_classes[curr_class].append(key)
        curr_int = 0
        for key in self.env.all_rewards.keys():
            if random_V_Q:
                self.Q[key] = random.uniform(0, 1)
            else:
                self.Q[key] = 0
            if map_tuples_to_int == 1:
                self.two_tuple_int_map[key] = curr_int
                self.int_two_tuple_map[curr_int] = key
                t_Q_vec[curr_int] = self.Q[key]
                curr_int += 1
        if map_tuples_to_int == 1:
            self.Q_vec = t_Q_vec
        if map_tuples_to_int == 1:
            t_Q_act_vec = np.zeros(shape=self.Q_vec.shape)
            t_Q_state_vec = np.zeros(shape=self.Q_vec.shape)
            for key in self.env.all_rewards.keys():
                idx = self.two_tuple_int_map[key]
                t_Q_state_vec[idx] = self.one_tuple_int_map[key[0]]
                t_Q_act_vec[idx] = self.one_tuple_int_map[key[1]]
        for key in self.env.all_actions.keys():
            greedy_action = self.get_action(state_tuple=key, Q_is_vec=0, force_greedy=1, epsilon_greedy=0, parse=1)
            t_policy[key] = greedy_action
            greedy_action = greedy_action
            self.greedy_Q_val[key] = [greedy_action, self.Q[(key, greedy_action)]]

        self.policy = t_policy
        if map_tuples_to_int == 1:
            self.Q_state_vec = t_Q_state_vec
            self.Q_act_vec = t_Q_act_vec
            t_Q_mat = np.full(shape=(len(self.env.all_actions.keys()), len(self.env.all_actions.keys())), fill_value=-np.inf)
            if map_tuples_to_int == 1:
                for key in self.env.all_rewards.keys():
                    i = self.one_tuple_int_map[key[0]]
                    j = self.one_tuple_int_map[key[1]]
                    t_Q_mat[i, j] = self.Q[key]
            self.Q_mat = t_Q_mat
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("AgentTools constructed in " + str(round(calc_time, 2)) + " seconds")

    def standard_learning_rate(self):
        return 1 / (1 + self.t)

    def get_action(self, Q_is_vec, state_tuple=None, force_greedy=1, epsilon_greedy=0, parse=0):
        if state_tuple:
            curr_state = state_tuple
        else:    # state == None
            curr_state = self.env.state
        if parse == 1:
            if not state_tuple:
                raise ValueError("function 'get_action must get a state if parse == 1")
            opt_key = -1
            opt_Q_val = -1
            possible_actions_in_state = self.env.all_actions[curr_state]
            for action in possible_actions_in_state:
                curr_key = (curr_state, action)
                if self.Q[curr_key] > opt_Q_val:
                    opt_key = curr_key
                    opt_Q_val = self.Q[curr_key]
            return opt_key[1]
        p = self.standard_learning_rate()
        is_random = bernoulli.rvs(p)
        if epsilon_greedy == 0 or is_random == 0:
            if Q_is_vec == 1:     # if vec == 1: greedy is always forced
                curr_state_int = self.one_tuple_int_map[curr_state]
                mask = self.Q_state_vec == curr_state_int
                mask = mask * self.Q_vec
                coded_best_state_action = np.argmax(mask)
                ret_val = self.int_one_tuple_map[self.Q_act_vec[coded_best_state_action]]
                return ret_val
            else:
                if force_greedy == 1:
                    ret_val = self.greedy_Q_val[curr_state][0]
                    return ret_val
                else:
                    return self.policy[curr_state]
        else:
            possible_actions_in_state = self.env.all_actions[curr_state]
            i = random.randint(0, len(possible_actions_in_state) - 1)
            return possible_actions_in_state[i]

    def get_reward(self, Q_is_vec, action=None):
        curr_state = self.env.state
        if action:
            chosen_action = action
        else:
            chosen_action = self.get_action(Q_is_vec)
        return self.env.all_rewards[(curr_state, chosen_action)]

    def get_next_state(self, Q_is_vec, action=None):
        if action:
            chosen_action = action
        else:
            chosen_action = self.get_action(Q_is_vec)
        return self.env.observe_next_state(chosen_action)

    def termination_check(self, old_Q_vec=None, detect_convergence=True, threshold=0.001, memo_bound=100):
        if detect_convergence:
            if not isinstance(old_Q_vec, np.ndarray):
                raise ValueError("Must supply old_Q_vec for detect convergence mode")
            max_improvement = np.max((self.Q_vec - old_Q_vec) / self.Q_vec)
            if max_improvement < threshold:
                self.times_without_improvement += 1
            if memo_bound < self.times_without_improvement or self.t >= self.horizon:
                return self.policy
            else:
                return 0
        else:
            if self.t < self.horizon:
                return 0
            else:
                return self.policy

    def general_TDT_learning_step(self, temporal_difference_target, is_epsilon_greedy=1):
        action = self.get_action(force_greedy=1, Q_is_vec=0, epsilon_greedy=is_epsilon_greedy)
        state = self.env.state
        alpha = self.standard_learning_rate()
        self.Q[(state, action)] = (1 - alpha) * self.Q[(state, action)] + alpha * temporal_difference_target
        if self.t not in self.log.keys():
            self.log[self.t] = []
        self.log[self.t].append((state, action))
        self.log[self.t].append(self.Q[(state, action)])
        if self.Q[(state, action)] > self.greedy_Q_val[state][1]:
            self.greedy_Q_val[state] = [action, self.Q[(state, action)]]
            self.policy[state] = action
        self.t += 1
        if self.t % 100000 == 0:
            print("Completed " + str(self.t / 1000) + " * 10^3 time steps")
        self.env.step(action)

    def SARSA_step(self, is_epsilon_greedy=1):
        reward = self.get_reward(Q_is_vec=0)
        next_state = self.get_next_state(Q_is_vec=0)
        next_action = self.get_action(state_tuple=next_state, Q_is_vec=0, epsilon_greedy=is_epsilon_greedy, force_greedy=1)
        temporal_difference_target = reward + self.discount_factor * self.Q[(next_state, next_action)]
        self.general_TDT_learning_step(temporal_difference_target, is_epsilon_greedy)
        to_end = self.termination_check(detect_convergence=False)
        if to_end != 0:
            return self.policy
        else:
            return 0

    def SARSA_lambda_step(self, is_epsilon_greedy=1, const_alpha=1, mat_Q=0, improvement_check=False):
        action = self.get_action(Q_is_vec=1, force_greedy=1, epsilon_greedy=is_epsilon_greedy)
        state = self.env.state
        reward = self.get_reward(Q_is_vec=1, action=action)
        next_state = self.get_next_state(Q_is_vec=1, action=action)
        next_action = self.get_action(Q_is_vec=1, state_tuple=next_state, force_greedy=1, epsilon_greedy=is_epsilon_greedy)
        concat_state = (state, action)
        next_concat_state = (next_state, next_action)
        if mat_Q == 1:
            delta = reward + self.discount_factor * self.Q_mat[self.one_tuple_int_map[next_state],self.one_tuple_int_map[next_action]] \
                                        - self.Q_mat[self.one_tuple_int_map[state], self.one_tuple_int_map[action]]
            self.e_mat[self.one_tuple_int_map[state], self.one_tuple_int_map[action]] += 1
            self.e_mat *= float(self.lambd * self.discount_factor)
            if const_alpha == 1:
                self.Q_mat += self.alpha * delta * self.e_mat
            else:
                self.Q_mat += self.standard_learning_rate() * delta * self.e_mat
        else:   # Vector mode
            delta = reward + self.discount_factor * self.Q_vec[self.two_tuple_int_map[next_concat_state]] - self.Q_vec[self.two_tuple_int_map[concat_state]]
            self.e_vec *= float(self.lambd * self.discount_factor)
            self.e_vec[self.two_tuple_int_map[concat_state]] += 1
            if improvement_check:
                old_Q_vec = self.Q_vec.copy()
            if const_alpha == 1:
                self.Q_vec += self.alpha * delta * self.e_vec
            else:
                self.Q_vec += self.standard_learning_rate() * delta * self.e_vec
            print_if_decreased = False
            if improvement_check and print_if_decreased:
                if np.any(old_Q_vec > self.Q_vec):
                    print("At time " + str(self.t) + " Q of some state action pair decreased")
        if self.t not in self.log.keys():
            self.log[self.t] = []
        self.log[self.t].append((state, action))
        self.log[self.t].append(self.Q[(state, action)])
        self.t += 1
        if self.t % 1000 == 0:
            print("Completed " + str(self.t / 1000) + " * 10^3 time steps")
        self.env.step(action)
        if improvement_check:
            to_end = self.termination_check(old_Q_vec=old_Q_vec)
        else:
            to_end = self.termination_check(detect_convergence=False)
        if to_end != 0:
            if mat_Q == 1:
                print("No mat support!!!!!!!")
            else:
                for i in range(self.Q_vec.shape[0]):
                    key = self.int_two_tuple_map[i]
                    self.Q[key] = self.Q_vec[i]
                return self.policy
        else:
            return 0

    def greedy_Q_action_considering_class(self, next_class):
        possible_actions_in_state = self.env.all_actions[self.env.state]
        opt_key = -1
        opt_Q_val = -1
        for action in possible_actions_in_state:
            if self.standard_state_classification(action) == next_class:  # action fits class
                curr_key = (self.env.state, action)
                if self.Q[curr_key] > opt_Q_val:
                    opt_key = curr_key
                    opt_Q_val = self.Q[curr_key]
        if isinstance(opt_key, int):  # No relevant state according to class
            return -1
        else:
            return opt_key[1]

    def move_to_random_state(self):
        i = random.randint(0, len(self.env.all_states) - 1)
        return self.env.all_states[i]

    def generate_audio_sequence(self, in_samples=0, max_history=8, enhance=1):
        ret_tuple = ()
        history = []
        if in_samples == 0:
            samples = self.horizon
        else:
            samples = in_samples
        for time_step in range(samples):
            if isinstance(self.env.state, str):
                raise ValueError("Policy learned illegal action")
            if self.policy[self.env.state] == self.env.state:
                print("State s == s' occurred in s = " + str(self.env.state))
                self.env.state = self.move_to_random_state()
            if self.env.state in history:
                print("state s in history occurred in s = " + str(self.env.state))
                self.env.state = self.move_to_random_state()

            if enhance == 1:
                mode = random.randint(0, 7)
                if mode < 2:
                    ret_tuple += self.env.state
                elif mode > 7:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                else:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state

            elif enhance == 2:
                if len(ret_tuple) % 16 == 12:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                elif len(ret_tuple) % 16 == 13:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                elif len(ret_tuple) % 16 == 14:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                elif len(ret_tuple) % 16 == 15:
                    ret_tuple += self.env.state
                else:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state

            else:
                ret_tuple += self.env.state
                ret_tuple += self.env.state
            if len(history) <= max_history:
                history += [self.env.state]
            else:
                temp_var = history.pop(0)
                history += [self.env.state]
            action = self.policy[self.env.state]
            self.env.step(action)
        return ret_tuple

    def get_random_state_from_class(self, desired_class):
        if desired_class in self.state_classes.keys():
            optional_states = self.state_classes[desired_class]
            idx = random.randint(0, len(optional_states) - 1)
            return optional_states[idx]
        else:
            return -1

    def fit_state_to_class(self, state, target_class, safe_mode=True):
        helping_list = []
        last_note = -1
        for idx, item in enumerate(state):
            if target_class[idx] == 999 or target_class[idx] == 0:  # Keep item - always for 1st in state
                helping_list.append(item)
                if item != 666:
                    last_note = item
                else:    # Item == 666
                    state_no_666 = [i for i in state if i < 128]
                    if len(state_no_666) == 0:
                        state_no_666 = [72]  # If C5 not good for the scale - change
                    last_note = int(statistics.mean(state_no_666))
            elif target_class[idx] == 666:  # Silence
                helping_list.append(666)
            else:   # We want to fix (target_class[idx] in [-1, 1])
                if target_class[idx] not in [-1, 1]:
                    raise ValueError("target_class[idx] is " + str(target_class[idx]) + " but it must be -1 or 1")
                if item == 666 and safe_mode:    # We got 666 that we try to 'fix' - need special treatment
                    options_high = [5, 7]
                    options_low = [-5, -7]
                    j = random.randint(0, len(options_high) - 1)
                    if last_note == 666:   # last_note is 666, we 'assume' last_note is C4
                        if target_class[idx] == 1:
                            added_item = 72 + options_high[j]
                        else:     # target_class[idx] == -1
                            added_item = 72 + options_low[j]
                        helping_list.append(added_item)
                    else:    # last_note is not 666
                        if target_class[idx] == 1 and last_note + options_high[j] < 128:
                            added_item = last_note + options_high[j]
                        elif target_class[idx] == -1 and last_note + options_low[j] >= 0:
                            added_item = last_note + options_low[j]
                        else:
                            added_item = last_note
                        helping_list.append(added_item)
                elif target_class[idx] == 1:
                    if item > last_note or (item + 12) > 127 or last_note == 666:
                        added_item = item
                        helping_list.append(added_item)
                    else:   # note lower than the last, and it's possible to increase it in an octave
                        added_item = item + 12
                        helping_list.append(added_item)
                elif target_class[idx] == -1:
                    if item < last_note or (item - 12) < 0 or last_note == 666:
                        added_item = item
                        helping_list.append(added_item)
                    else:   # note higher than the last, and it's possible to decrease it in an octave.
                        added_item = item - 12
                        helping_list.append(added_item)
                else:
                    print("Logically impossible option occured in 'fit_state_to_class'")
                if added_item > 127:
                    raise ValueError("The function added a note that's higher than 127 !!!")
                last_note = added_item
        return tuple(helping_list)


    def fix_audio(self, up_down_feature_lst, method=1, max_history=8):
        ret_tuple = ()
        len_of_env_state = len(self.env.state)

        initial_class = self.standard_state_classification(tuple(up_down_feature_lst[:len_of_env_state]))
        try_getting_random_state = self.get_random_state_from_class(initial_class)
        if not isinstance(try_getting_random_state, int):
            self.env.state = try_getting_random_state
        else:
            self.env.state = self.move_to_random_state()
        for idx, item in enumerate(up_down_feature_lst):
            if idx % len_of_env_state == 0 and (idx - 1 + len_of_env_state) < len(up_down_feature_lst) and method in [0, 1]: # Condition to generate new state
                if 999 not in up_down_feature_lst:
                    print("Value 999 isn't in 'up_down_feature_lst' but this value is expected in mode 1,0...")
                if idx == 0:
                    last_added_tuple = self.env.state
                    ret_tuple += last_added_tuple
                else:
                    target_class = self.standard_state_classification(tuple(up_down_feature_lst[idx:idx + len_of_env_state]))
                    try_class_greedy = self.greedy_Q_action_considering_class(target_class)
                    if method == 0:
                        if isinstance(try_class_greedy, int):     # No possible next state that fits target_class
                            try_getting_random_state = self.get_random_state_from_class(target_class)
                            if isinstance(try_getting_random_state, int):
                                last_added_tuple = self.move_to_random_state()
                                self.env.state = last_added_tuple
                                ret_tuple += last_added_tuple
                            else:
                                last_added_tuple = try_getting_random_state
                                self.env.state = last_added_tuple
                                ret_tuple += last_added_tuple
                        else:
                            last_added_tuple = try_class_greedy
                            self.env.state = last_added_tuple
                            ret_tuple += last_added_tuple   # The class greedy attempt succeeded
                    elif method == 1:    # New method - best from methods [1-7]
                        if isinstance(try_class_greedy, int):     # No possible next state that fits target_class
                            next_state_greedy_no_class_consideration = self.get_action(
                                Q_is_vec=0, state_tuple=self.env.state, force_greedy=1, epsilon_greedy=0, parse=1)
                            corrected_state = self.fit_state_to_class(
                                next_state_greedy_no_class_consideration, target_class)
                            last_added_tuple = corrected_state
                            ret_tuple += last_added_tuple
                        else:
                            last_added_tuple = try_class_greedy
                            self.env.state = last_added_tuple
                            ret_tuple += last_added_tuple  # The class greedy attempt succeeded
                    else:
                        print("Value of 'method' is incorrect")
        if method == 2:
            if 999 in up_down_feature_lst:
                print("Value 999 appears in 'up_down_feature_lst' but this value is not expected in mode 2")
            curr_state = self.env.state
            desired_length = len(up_down_feature_lst)
            iterations = int(desired_length / len(curr_state))
            for i in range(iterations):
                next_state = self.get_action(Q_is_vec=False, state_tuple=curr_state)
                ret_tuple += next_state
                curr_state = next_state
        elif method == 3:
            if 999 in up_down_feature_lst:
                print("Value 999 appears in 'up_down_feature_lst' but this value is not expected in mode 3")
            curr_state = self.env.state
            desired_length = len(up_down_feature_lst)
            iterations = int(desired_length / len(curr_state))
            for i in range(iterations):
                next_state = self.get_action(Q_is_vec=False, state_tuple=curr_state)
                ret_tuple += next_state
                curr_state = next_state
            ret_list = list(ret_tuple)
            for item in ret_list:
                if item not in [666, 0]:
                    last_note = item
                    break
            for idx, item in enumerate(ret_list):
                if up_down_feature_lst[idx] == 666:
                    ret_list[idx] = 666
                elif up_down_feature_lst[idx] == 0:
                    ret_list[idx] = last_note
                elif up_down_feature_lst[idx] == 1:
                    if item <= last_note:
                        if item + 12 < 128:
                            ret_list[idx] += 12
                    last_note = ret_list[idx]
                elif up_down_feature_lst[idx] == -1:
                    if item >= last_note:
                        if item - 12 > -1:
                            ret_list[idx] += -12
                    last_note = ret_list[idx]
        elif method == 4:
            if 999 in up_down_feature_lst:
                print("Value 999 appears in 'up_down_feature_lst' but this value is not expected in mode 4")
            curr_state = self.env.state
            desired_length = len(up_down_feature_lst)
            iterations = int(desired_length / len(curr_state))
            for i in range(iterations):
                possible_actions_in_state = self.env.all_actions[curr_state]
                i = random.randint(0, len(possible_actions_in_state) - 1)
                next_state = possible_actions_in_state[i]
                ret_tuple += next_state
                if curr_state == next_state:
                    curr_state = self.move_to_random_state()
                else:
                    curr_state = next_state
        elif method == 7:   # Generate only according to the observations - old
            curr_state = self.env.state
            desired_length = len(up_down_feature_lst)
            iterations = int(desired_length / len(curr_state))
            for i in range(iterations):
                possible_actions = self.env.all_states
                j = random.randint(0, len(possible_actions) - 1)
                ret_tuple += possible_actions[j]
        elif method == 6:  # Generate only according to the observations - new
            curr_state = self.env.state
            desired_length = len(up_down_feature_lst)
            iterations = int(desired_length / len(curr_state))
            for i in range(iterations):
                j = random.randint(0, len(tuple(self.Q.keys())) - 1)
                ret_tuple += tuple(self.Q.keys())[j][0]
        else:
            if method not in [0, 1]:
                print("Value of 'method' is incorrect")
        return ret_tuple

    def dump_Q(self, file_name, method=1):
        if method == 0:
            helping_dict = {}
            for key in self.Q.keys():
                helping_dict[key] = [self.Q[key]]
            helping_df = pd.DataFrame.from_dict(helping_dict)
            helping_df.to_csv("Q_func_out.csv", sep=",")
        else:     # method == 1
            helping_lst = []
            for key in self.Q.keys():
                temp_lst = list(key[0]) + list(key[1]) + [self.Q[key]]
                helping_lst.append(temp_lst)
            arr = np.array(helping_lst)
            np.savetxt(file_name, arr, delimiter=",")

    def read_Q_csv(self, file_name, len_state, len_action):
        print("CSV read began")
        helping_arr = np.loadtxt(file_name, delimiter=",")
        for i in range(helping_arr.shape[0]):
            elem0 = []
            elem1 = []
            for j in range(helping_arr.shape[1]):
                if j < len_state:
                    elem0.append(helping_arr[i, j])
                elif j < (len_state + len_action):
                    elem1.append(helping_arr[i, j])
                else:   # last item in the row
                    self.Q[(tuple(elem0), tuple(elem1))] = helping_arr[i, j]
        print("Q constructed")