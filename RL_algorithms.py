import numpy as np
import random
import time
import statistics
from scipy.stats import bernoulli
import gymnasium as gym
from gymnasium import spaces
import torch
from scipy.stats import entropy



# Define the class of gym env
class FiniteHorizonDDPEnv(gym.Env):
    def __init__(self, states, actions, rewards, horizon):
        super(FiniteHorizonDDPEnv, self).__init__()

        self.states = states  # List of possible states (integers)
        self.actions = actions  # Dict: state -> list of valid next states
        self.rewards = rewards  # Dict: (state, action) -> reward
        self.horizon = horizon  # Finite horizon

        self.state = None  # Current state
        self.timestep = 0  # Track episode length

        self.observation_space = spaces.Discrete(len(states))
        self.action_space = spaces.Discrete(max(len(a) for a in actions.values()))

    def reset(self, seed=None, options=None, random_reset=True):
        super().reset(seed=seed)
        self.state = self.states[0]  # Reset to initial state (assumed first in list)
        if random_reset:
            idx = random.randint(0, len(self.states) - 1)
            self.state = self.states[idx]
        self.timestep = 0
        return self.state, {}

    def step(self, action):
        step_initial_state = self.state
        if action not in self.actions.get(self.state, []):
            done = self.timestep >= self.horizon
            return self.state, -10.0, done, False, {}  # Invalid action penalty

        next_state = action  # Action represents the next state directly
        reward = self.rewards.get((self.state, action), 0.0)

        self.state = next_state
        self.timestep += 1
        done = self.timestep >= self.horizon

        if next_state == step_initial_state:
            reward = -1.0   # penalty for choosing the same action again

        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.timestep}, State: {self.state}")


# Function to get the top 3 best actions for a given state
def get_top_actions(model, state, top_n=3):
    obs = torch.tensor([state], dtype=torch.float32).to(device='cuda')  # Convert state to tensor
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)  # Get action distribution
        action_probs = torch.exp(dist.distribution.logits).cpu().numpy()  # Convert to probabilities

    # Get top N actions by sorting
    top_actions = np.argsort(action_probs[0])[::-1][:top_n]
    return top_actions


def calculate_entropy(signal):
    # Make sure the signal is a np array
    signal = np.array(signal)

    # Get probability distribution
    values, counts = np.unique(signal, return_counts=True)
    probabilities = counts / counts.sum()

    return entropy(probabilities, base=2)  # Base 2 for bits unit


class DeterministicEnvTools:
    def __init__(self, states_list, actions_dict, rewards_dict, transitions_dict, initial_state):
        self.all_states = states_list         # All possible states (that were observed) in a list
        self.all_actions = actions_dict       # Dict in the form of: {state: [actions]}
        self.all_rewards = rewards_dict       # Dict in the form of: {(state, action): reward}
        self.transitions = transitions_dict   # Dict in the form of: {(state, action): nextstate}
        self.initial_state = initial_state
        self.state = initial_state

    def construct_EnvTools_from_observations_dict(self, observations_dict, arcs_for_state=5):
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


        vec_int_map = {}
        int_vec_map = {}

        curr_int = 0
        for item in self.all_states:
            vec_int_map[item] = curr_int
            int_vec_map[curr_int] = item
            curr_int += 1

        self.vec_int_map = vec_int_map
        self.int_vec_map = int_vec_map
        self.next_available_idx = curr_int

        self.actions_dict_new = {}

        for key in self.all_actions.keys():
            self.actions_dict_new[vec_int_map[key]] = []
            for item in self.all_actions[key]:
                self.actions_dict_new[vec_int_map[key]].append(vec_int_map[item])

        self.rewards_dict_new = {}

        for key in self.all_rewards.keys():
            new_key = (vec_int_map[key[0]], vec_int_map[key[1]])
            self.rewards_dict_new[new_key] = self.all_rewards[key]

        timer_end = time.time()
        calc_time = timer_end - timer_start

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

    def construct_AgentTools_from_env(self, map_tuples_to_int=1, random_V_Q=1):
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
