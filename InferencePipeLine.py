import os
import random
import time
import MIDI_IO as io
import MIDI_coding as code
import RL_algorithms as RL
from RL_algorithms import FiniteHorizonDDPEnv, get_top_actions, calculate_entropy
import audio_metrics as metrics
from stable_baselines3 import PPO
from prettytable import PrettyTable
import re
import sys
import mido
from copy import deepcopy

# Configurations
reference_file = "piano1_ref.mid"
file_to_fix = "single_notes_errors_piano_and_drums6.mid"
correct_file = "single_notes_piano_and_drums6.mid"
gen_or_fix = "fix"
model_to_use = "ppo_finite_horizon_500_steps_500000"
path_to_midi_files = "midi_files"

# Hyper-parameters
len_of_state = 4
top_n = 10   # Number of items to take after for generating the final policy
top_n_untrained = top_n
arcs_for_state = 10   # Minimal number of actions that will be legal for each state
horizon = 5e2
agent_time_steps = 5e1
steps_in_the_final_generation = 100
leading_items_to_remove_from_action_options = 0
comparison_loop_iterations = 100

# Extract audio and construct tools
fix_lst = io.vectorize_MIDI(file_to_fix, channel_filtering=0)
single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item, len_of_state=len_of_state) for item in single_notes_lst]

ref_lst = io.vectorize_MIDI(reference_file, channel_filtering=0)
ref_data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in ref_lst]

env_tools = RL.DeterministicEnvTools([], {}, {}, {}, -1)
env_tools.construct_EnvTools_from_observations_dict(ref_data[0], arcs_for_state=arcs_for_state)

agent_tools = RL.AgentTools(env_tools, horizon=horizon)
agent_tools.construct_AgentTools_from_env()

actions_dict = env_tools.actions_dict_new
rewards_dict = env_tools.rewards_dict_new
vec_int_map = env_tools.vec_int_map
int_vec_map = env_tools.int_vec_map
all_states = [vec_int_map[item] for item in env_tools.all_states]
next_available_idx = env_tools.next_available_idx


# Initialize gym
env = FiniteHorizonDDPEnv(
    states=all_states,
    actions=actions_dict,
    rewards=rewards_dict,
    horizon=horizon
)

# Load the trained model
model = PPO.load(model_to_use)

# Load the untrained model
untrained_model = PPO.load("ppo_finite_horizon_not_trained")


# Extract the policy learned by the agent (PPO)
timer_start = time.time()
print("Extracting policies from PPO")

policy_dict_ppo = {}
policy_dict_ppo_untrained = {}
for state in all_states:
    actions = get_top_actions(model=model, state=state, top_n=top_n)
    policy_dict_ppo[state] = actions
    actions_untrained = get_top_actions(model=untrained_model, state=state, top_n=top_n_untrained)
    policy_dict_ppo_untrained[state] = actions_untrained

vec_policy_ppo = {}
vec_policy_ppo_untrained = {}
for key in policy_dict_ppo.keys():
    new_lst = [int_vec_map[item] for item in policy_dict_ppo[key]]
    vec_policy_ppo[int_vec_map[key]] = new_lst
    new_lst_untrained = [int_vec_map[item] for item in policy_dict_ppo_untrained[key]]
    vec_policy_ppo_untrained[int_vec_map[key]] = new_lst_untrained

timer_end = time.time()

print(f"Policies extracted in {round(timer_end - timer_start, 2)} seconds")

# Generate audio with trained and untrained policies
print("Generating audio by PPO")
timer_start = time.time()
generated_tuple = ()
idx = random.randint(0, len(list(vec_policy_ppo.keys())) - 1)
s = list(vec_policy_ppo.keys())[idx]
for _ in range(steps_in_the_final_generation):
    options_lst = vec_policy_ppo[s]
    idx = random.randint(leading_items_to_remove_from_action_options, len(options_lst) - 1)
    s = vec_policy_ppo[s][idx]
    generated_tuple += s

decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple)
timer_end = time.time()
print(f"Audio generated in {round(timer_end - timer_start, 2)} seconds by PPO")


# Audio generation params
offset = -12  # Semi-tone shifts up or down to the entire generated audio
target_track = 0
target_channel = 3
file_name_for_inference = "solo and back.MID"
p = 1 / 3   # Probability to change a note (error)
error_range = 2   # Range of error

# Inference
print("Starting inference run")
mid = mido.MidiFile(file_name_for_inference)
mid_errors = deepcopy(mid)
mid_fixed = deepcopy(mid)
pointer_on_generated_tuple = 0
for i, track in enumerate(mid.tracks):
    for j, msg in enumerate(track):
        if msg.type in ['note_on', 'note_off'] and msg.channel == target_channel:
            to_cause_error = random.randint(0, int(1 / p))
            if to_cause_error == 0:
                to_cause_error = 1
            else:
                to_cause_error = 0
            error_size = random.randint(-error_range, error_range) * to_cause_error
            fixed_note = offset + generated_tuple[pointer_on_generated_tuple]
            msg_to_errors = deepcopy(msg)
            msg_to_fixed = deepcopy(msg)
            msg_to_errors.note += error_size
            msg_to_fixed.note = fixed_note
            mid_errors.tracks[i][j] = msg_to_errors
            mid_fixed.tracks[i][j] = msg_to_fixed

mid_errors.save('inference_output_errors.mid')
mid_fixed.save('inference_output_fixed.mid')
print("Inference succeed files saved")




