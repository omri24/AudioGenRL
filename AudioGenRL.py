import random

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import MIDI_IO as io
import MIDI_coding as code
import RL_algorithms as RL
from stable_baselines3 import PPO

# Files and parameters
reference_file = "piano1.mid"
file_to_fix = "single_notes_errors_piano_and_drums2.mid"
correct_file = "single_notes_piano_and_drums2.mid"
gen_or_fix = "GEN"

top_n = 10   # Number of items to take after for generating the final policy
arcs_for_state = 5   # Number of actions that will be legal for each state
horizon = 100
PPO_time_steps = 100
steps_in_the_final_generation = horizon
leading_items_to_remove_from_action_options = 5

# Extract audio
fix_lst = io.vectorize_MIDI(file_to_fix, channel_filtering=0)
single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item, len_of_state=4) for item in single_notes_lst]

ref_lst = io.vectorize_MIDI(reference_file, channel_filtering=0)
ref_data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in ref_lst]

my_env = RL.DeterministicEnvTools([], {}, {}, {}, -1)
my_env.construct_env_from_observations_dict(ref_data[0], arcs_for_state=arcs_for_state)

agent = RL.AgentTools(my_env, horizon=horizon)
agent.construct_agent_from_env()

actions_dict = my_env.actions_dict_new
rewards_dict = my_env.rewards_dict_new
vec_int_map = my_env.vec_int_map
int_vec_map = my_env.int_vec_map
all_states = [vec_int_map[item] for item in my_env.all_states]

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
            return self.state, -1.0, done, False, {}  # Invalid action penalty

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

# Initialize gym
env = FiniteHorizonDDPEnv(
    states=all_states,
    actions=actions_dict,
    rewards=rewards_dict,
    horizon=horizon
)

# Train agent using PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=PPO_time_steps)

# Save trained model
model.save("ppo_finite_horizon")
print("Model trained and saved successfully!")

# Load the trained model
model = PPO.load("ppo_finite_horizon")

# Function to get the top 3 best actions for a given state
def get_top_actions(model, state, top_n=3):
    obs = torch.tensor([state], dtype=torch.float32).to(device='cuda')  # Convert state to tensor
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)  # Get action distribution
        action_probs = torch.exp(dist.distribution.logits).cpu().numpy()  # Convert to probabilities

    # Get top N actions by sorting
    top_actions = np.argsort(action_probs[0])[::-1][:top_n]
    return top_actions


print("Extracting policy")

policy_dict = {}

for state in all_states:
    actions = get_top_actions(model=model, state=state, top_n=top_n)
    policy_dict[state] = actions

vec_policy = {}
for key in policy_dict.keys():
    new_lst = [int_vec_map[item] for item in policy_dict[key]]
    vec_policy[int_vec_map[key]] = new_lst

print("Policy extracted")

if gen_or_fix == "GEN":
    generated_tuple = ()
    s = list(vec_policy.keys())[0]
    for _ in range(steps_in_the_final_generation):
        options_lst = vec_policy[s]
        idx = random.randint(leading_items_to_remove_from_action_options, len(options_lst) - 1)
        s = vec_policy[s][idx]    # Select last item, could be other items
        generated_tuple += s
    decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple)
    n = io.export_MIDI([decoded_generated_tuple], file_name="out_new.mid", ticks_per_sixteenth=180)

if gen_or_fix == "FIX":
    None





"""
print("Extracting policy")
policy_dict = {}

for state in all_states:
    action, _ = model.predict(state, deterministic=True)  # Get optimal action
    policy_dict[state] = action

vec_policy = {}
for key in policy_dict.keys():
    vec_policy[int_vec_map[key]] = int_vec_map[policy_dict[key]]








def train_agent_random(env, episodes=100):
    policy = {}  # State -> best action mapping
    returns = {}  # Track returns for each (state, action) pair

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_history = []

        while not done:
            valid_actions = env.actions[state]
            action = random.choice(valid_actions)  # Choose randomly among valid actions
            next_state, reward, done, _ = env.step(action)
            episode_history.append((state, action, reward))
            state = next_state

        G = 0  # Return
        for state, action, reward in reversed(episode_history):
            G += reward
            if (state, action) not in returns:
                returns[(state, action)] = []
            returns[(state, action)].append(G)
            policy[state] = max(env.actions[state], key=lambda a: np.mean(returns.get((state, a), [0])))

    return policy














class FiniteHorizonDDPEnv(gym.Env):
    def __init__(self, actions_dict, rewards_dict, horizon=10):
        super().__init__()
        self.horizon = horizon  # Number of time steps before termination
        self.current_step = 0

        # Define state space (4-element arrays with integers between 0 and 127)
        self.observation_space = gym.spaces.Box(low=0, high=127, shape=(4,), dtype=np.int32)

        # Define possible actions for each state
        self.actions_dict = actions_dict

        # Define rewards for each (state, action) pair
        self.rewards_dict = rewards_dict

        # Find the maximum number of actions for any state
        max_actions = max(len(actions) for actions in self.actions_dict.values())

        # Define discrete action space with a fixed number of possible actions
        self.action_space = gym.spaces.Discrete(max_actions)
        self.state = np.array(list(actions_dict.keys())[0], dtype=np.int32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.state = np.array([0, 0, 0, 0], dtype=np.int32)  # Start state
        return self.state, {}

    def step(self, action):
        state_tuple = tuple(self.state.tolist())

        if state_tuple in self.actions_dict:
            valid_actions = self.actions_dict[state_tuple]

            if action < len(valid_actions):  # Convert action index to actual action
                selected_action = valid_actions[action]
                reward = self.rewards_dict.get((state_tuple, selected_action), 0)
                self.state = np.roll(self.state, -1)
                self.state[-1] = selected_action
            else:
                reward = -10  # Penalize invalid action
        else:
            reward = -10  # Penalize if the state has no valid actions

        self.current_step += 1
        done = self.current_step >= self.horizon
        return self.state, reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        pass



"""