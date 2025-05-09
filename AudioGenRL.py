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
from stable_baselines3.common.callbacks import BaseCallback


# Configurations
reference_file = "piano1_ref.mid"
file_to_fix = "single_notes_errors_piano_and_drums6.mid"
correct_file = "single_notes_piano_and_drums6.mid"
gen_or_fix = "fix"
model_to_use = "ppo_finite_horizon_500_steps_500000"
train_PPO = True
path_to_midi_files = "midi_files"
snap_shots = 50

# Hyper-parameters
len_of_state = 4
top_n = 10   # Number of items to take after for generating the final policy
top_n_untrained = top_n
arcs_for_state = 10   # Minimal number of actions that will be legal for each state
horizon = 500   # Leave this 500, don't change
agent_time_steps = 1e4
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

if train_PPO:
    # Train agent using PPO
    print("Model training with PPO started")
    timer_start = time.time()

    if snap_shots > 0:
        # Calculate when to save a snapshot of the model - to further evaluation
        snap_shot_delta = int(agent_time_steps / snap_shots)

        class SnapShotPPO(BaseCallback):
            def __init__(self, save_freq, save_path, verbose=0):
                super(SnapShotPPO, self).__init__(verbose)
                self.save_freq = save_freq
                self.save_path = save_path
                os.makedirs(self.save_path, exist_ok=True)

            def _on_step(self) -> bool:
                if self.num_timesteps % self.save_freq == 0:
                    save_file = os.path.join(self.save_path, f'model_{self.num_timesteps}_steps')
                    self.model.save(save_file)
                    if self.verbose:
                        print(f"Saved model at {self.num_timesteps} timesteps to {save_file}")
                return True

        model = PPO("MlpPolicy", env, verbose=1)
        save_callback = SnapShotPPO(save_freq=snap_shot_delta, save_path="./ppo_checkpoints", verbose=1)
        model.learn(total_timesteps=agent_time_steps, callback=save_callback)

    else:   # No snap-shots
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=agent_time_steps)

        # Save trained model
        model.save(f"ppo_finite_horizon_{int(horizon)}_steps_{int(agent_time_steps)}")

    timer_end = time.time()
    print(f"Model trained and saved in {round(timer_end - timer_start, 2)} seconds")
    sys.exit(0)

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
if gen_or_fix.lower() == "gen":
    print("Generating audio")
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
    n = io.export_MIDI([decoded_generated_tuple], file_name="out_new.mid", ticks_per_sixteenth=180)
    timer_end = time.time()
    print(f"Audio generated in {round(timer_end - timer_start, 2)} seconds")

    # Generate with untrained model
    generated_tuple_untrained = ()
    idx = random.randint(0, len(list(vec_policy_ppo_untrained.keys())) - 1)
    s = list(vec_policy_ppo_untrained.keys())[idx]
    for _ in range(steps_in_the_final_generation):
        options_lst = vec_policy_ppo_untrained[s]
        idx = random.randint(0, len(options_lst) - 1)
        s = vec_policy_ppo_untrained[s][idx]
        generated_tuple_untrained += s

# Fix errors
elif gen_or_fix.lower() == "fix":
    print("Fixing audio")
    times_trained_policy_used_action_not_from_training_data = 0
    timer_start = time.time()
    generated_tuple = ()
    lst_for_entropy = []
    idx = random.randint(0, len(list(vec_policy_ppo.keys())) - 1)
    s = list(vec_policy_ppo.keys())[idx]
    up_down_feature_lst = up_down_feature_lst_lst[0]
    steps_in_final_fix = int(len(up_down_feature_lst) / len_of_state)
    for i in range(steps_in_final_fix):
        options_lst = vec_policy_ppo[s]
        idx = random.randint(leading_items_to_remove_from_action_options, len(options_lst) - 1)
        s = vec_policy_ppo[s][idx]
        target_class = up_down_feature_lst[4 * i: 4 * i + len_of_state]
        s_to_add = agent_tools.fit_state_to_class(s, target_class)
        generated_tuple += s_to_add
        if s_to_add not in vec_int_map.keys():
            vec_int_map[s_to_add] = next_available_idx
            int_vec_map[next_available_idx] = s_to_add
            next_available_idx += 1
            times_trained_policy_used_action_not_from_training_data += 1
        lst_for_entropy.append(vec_int_map[s_to_add])

    decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple)
    steps_regex = re.search("steps_\d+", model_to_use)
    horizon_regex = re.search("horizon_\d+", model_to_use)
    steps_for_name = model_to_use[steps_regex.start() + 6: steps_regex.end()]
    horizon_for_name = model_to_use[horizon_regex.start() + 8: horizon_regex.end()]
    n = io.export_MIDI([decoded_generated_tuple], file_name=f"Generated_audio_horizon_{horizon_for_name}_steps_{steps_for_name}.mid", ticks_per_sixteenth=180)
    timer_end = time.time()
    print(f"Audio fixed in {round(timer_end - timer_start, 2)} seconds")
    print(f"In the audio generation, the policy used action not from the training data {times_trained_policy_used_action_not_from_training_data} times")

    # Fix with untrained model
    generated_tuple_untrained = ()
    lst_for_entropy_untrained = []
    idx = random.randint(0, len(list(vec_policy_ppo_untrained.keys())) - 1)
    s = list(vec_policy_ppo_untrained.keys())[idx]
    up_down_feature_lst = up_down_feature_lst_lst[0]
    steps_in_final_fix = int(len(up_down_feature_lst) / len_of_state)
    for i in range(steps_in_final_fix):
        options_lst = vec_policy_ppo_untrained[s]
        idx = random.randint(0, len(options_lst) - 1)
        s = vec_policy_ppo_untrained[s][idx]
        target_class = up_down_feature_lst[4 * i: 4 * i + len_of_state]
        s_to_add = agent_tools.fit_state_to_class(s, target_class)
        generated_tuple_untrained += s_to_add
        if s_to_add not in vec_int_map.keys():
            vec_int_map[s_to_add] = next_available_idx
            int_vec_map[next_available_idx] = s_to_add
            next_available_idx += 1
        lst_for_entropy_untrained.append(vec_int_map[s_to_add])

else:
    generated_tuple = None
    lst_for_entropy = None
    lst_for_entropy_untrained = None
    times_trained_policy_used_action_not_from_training_data = 0
    raise ValueError(f"Unsupported value for 'gen or fix': {gen_or_fix}")



if gen_or_fix.lower() == "fix":
    # Compare trained and untrained versions
    generated_tuple_trained = generated_tuple
    lst_for_entropy_trained = lst_for_entropy

    # Load the 'correct' audio
    correct_lst = io.vectorize_MIDI(correct_file, channel_filtering=0)
    single_notes_lst_corrected = [code.get_single_note_audio_from_multi_note_audio(item) for item in correct_lst]
    encoded_correct = [code.single_note_modulo_encoder(item) for item in single_notes_lst_corrected]

    # Calculate the improvement
    shortest_sequence_len = min(len(encoded_correct[0]), len(generated_tuple_trained), len(generated_tuple_untrained))
    # other dict option dist_dict={0: 0, 1: 5, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6, 7: 1, 8: 4, 9: 3, 10: 2, 11: 5}
    delta_correct_untrained = metrics.general_vector_modulo_12_metric(
        list(encoded_correct[0])[:shortest_sequence_len], list(generated_tuple_untrained)[:shortest_sequence_len])
    delta_correct_trained = metrics.general_vector_modulo_12_metric(
        list(encoded_correct[0])[:shortest_sequence_len], list(generated_tuple_trained)[:shortest_sequence_len])
    mean_delta_untrained = delta_correct_untrained / shortest_sequence_len
    mean_delta_trained = delta_correct_trained / shortest_sequence_len
    print("\nComparing the performance of the trained model and the untrained model:")
    print(f"Mean delta[correct, untrained]: {round(mean_delta_untrained, 2)}")
    print(f"Mean delta[correct, trained]: {round(mean_delta_trained, 2)}")

    # Positive value of 'improvement' is what we want
    improvement = (mean_delta_untrained - mean_delta_trained) / mean_delta_untrained
    print(f"Improvement (%): {round(improvement * 100, 2)}")

    # Check the entropy of the generated audio
    entropy_untrained = calculate_entropy(lst_for_entropy_untrained)
    entropy_trained = calculate_entropy(lst_for_entropy_trained)
    print(f"Entropy of the audio generated by the untrained model: {round(entropy_untrained, 2)}")
    print(f"Entropy of the audio generated by the trained model: {round(entropy_trained, 2)}")

    results_table = PrettyTable(["Iteration", "Mean delta[correct, untrained]", "Mean delta[correct, trained]",
                                "Improvement (%)", "Entropy untrained", "Entropy trained", "MIDI file"])

    # Get MIDI files
    midi_files = os.listdir(path_to_midi_files)
    midi_files_errors = [os.path.join(path_to_midi_files, item) for item in midi_files if re.search("_errors_", item)]
    midi_files_correct = [item[:re.search("_errors", item).start()] + item[re.search("_errors", item).end():] for item in midi_files_errors]
    midi_files_errors_correct = tuple(zip(midi_files_errors, midi_files_correct))
    midi_files_names = [item[1][re.search("single_notes", item[1]).start():len(item[1]) - 4] for item in midi_files_errors_correct]

    # Extract relevant objects
    up_down_feature_lst_and_encoded_correct = []
    for file_to_fix, correct_file in midi_files_errors_correct:

        # Extract audio and construct tools
        fix_lst = io.vectorize_MIDI(file_to_fix, channel_filtering=0)
        single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
        up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item, len_of_state=len_of_state) for item in single_notes_lst]

        # Load the 'correct' audio
        correct_lst = io.vectorize_MIDI(correct_file, channel_filtering=0)
        single_notes_lst_corrected = [code.get_single_note_audio_from_multi_note_audio(item) for item in correct_lst]
        encoded_correct = [code.single_note_modulo_encoder(item) for item in single_notes_lst_corrected]
        up_down_feature_lst_and_encoded_correct.append((up_down_feature_lst_lst[0], encoded_correct))

    # Test the model in iterations
    num_of_midi_files = len(up_down_feature_lst_and_encoded_correct)
    for iteration in range(comparison_loop_iterations):
        audio_idx_to_use = iteration % num_of_midi_files
        up_down_feature_lst, encoded_correct = up_down_feature_lst_and_encoded_correct[audio_idx_to_use]

        # Fix errors
        generated_tuple = ()
        lst_for_entropy = []
        idx = random.randint(0, len(list(vec_policy_ppo.keys())) - 1)
        s = list(vec_policy_ppo.keys())[idx]
        steps_in_final_fix = int(len(up_down_feature_lst) / len_of_state)
        for i in range(steps_in_final_fix):
            options_lst = vec_policy_ppo[s]
            idx = random.randint(leading_items_to_remove_from_action_options, len(options_lst) - 1)
            s = vec_policy_ppo[s][idx]
            target_class = up_down_feature_lst[4 * i: 4 * i + len_of_state]
            s_to_add = agent_tools.fit_state_to_class(s, target_class)
            generated_tuple += s_to_add
            if s_to_add not in vec_int_map.keys():
                vec_int_map[s_to_add] = next_available_idx
                int_vec_map[next_available_idx] = s_to_add
                next_available_idx += 1
            lst_for_entropy.append(vec_int_map[s_to_add])

        # Fix with untrained model
        generated_tuple_untrained = ()
        lst_for_entropy_untrained = []
        idx = random.randint(0, len(list(vec_policy_ppo_untrained.keys())) - 1)
        s = list(vec_policy_ppo_untrained.keys())[idx]
        steps_in_final_fix = int(len(up_down_feature_lst) / len_of_state)
        for i in range(steps_in_final_fix):
            options_lst = vec_policy_ppo_untrained[s]
            idx = random.randint(0, len(options_lst) - 1)
            s = vec_policy_ppo_untrained[s][idx]
            target_class = up_down_feature_lst[4 * i: 4 * i + len_of_state]
            s_to_add = agent_tools.fit_state_to_class(s, target_class)
            generated_tuple_untrained += s_to_add
            if s_to_add not in vec_int_map.keys():
                vec_int_map[s_to_add] = next_available_idx
                int_vec_map[next_available_idx] = s_to_add
                next_available_idx += 1
            lst_for_entropy_untrained.append(vec_int_map[s_to_add])

        # Compare trained and untrained versions
        generated_tuple_trained = generated_tuple
        lst_for_entropy_trained = lst_for_entropy

        # Calculate the improvement
        shortest_sequence_len = min(len(encoded_correct[0]), len(generated_tuple_trained), len(generated_tuple_untrained))
        # other dict option dist_dict={0: 0, 1: 5, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6, 7: 1, 8: 4, 9: 3, 10: 2, 11: 5}
        delta_correct_untrained = metrics.general_vector_modulo_12_metric(
            list(encoded_correct[0])[:shortest_sequence_len], list(generated_tuple_untrained)[:shortest_sequence_len])
        delta_correct_trained = metrics.general_vector_modulo_12_metric(
            list(encoded_correct[0])[:shortest_sequence_len], list(generated_tuple_trained)[:shortest_sequence_len])
        mean_delta_untrained = delta_correct_untrained / shortest_sequence_len
        mean_delta_trained = delta_correct_trained / shortest_sequence_len

        # Positive value of 'improvement' is what we want
        improvement = (mean_delta_untrained - mean_delta_trained) / mean_delta_untrained

        # Check the entropy of the generated audio
        entropy_untrained = calculate_entropy(lst_for_entropy_untrained)
        entropy_trained = calculate_entropy(lst_for_entropy_trained)

        results_table.add_row([iteration + 1, round(mean_delta_untrained, 2), round(mean_delta_trained, 2),
                              round(improvement * 100, 2), round(entropy_untrained, 2), round(entropy_trained, 2),
                               midi_files_names[audio_idx_to_use]])

    # Find averages
    mean_d_correct_untrained = [float(row[1]) for row in results_table._rows]
    mean_d_correct_trained = [float(row[2]) for row in results_table._rows]
    improvement_col = [float(row[3]) for row in results_table._rows]
    entropy_untrained_col = [float(row[4]) for row in results_table._rows]
    entropy_trained_col = [float(row[5]) for row in results_table._rows]

    d_correct_untrained_avg = round(sum(mean_d_correct_untrained) / len(mean_d_correct_untrained), 2)
    d_correct_trained_avg = round(sum(mean_d_correct_trained) / len(mean_d_correct_trained), 2)
    improvement_avg = round(sum(improvement_col) / len(improvement_col), 2)
    entropy_untrained_avg = round(sum(entropy_untrained_col) / len(entropy_untrained_col), 2)
    entropy_trained_avg = round(sum(entropy_trained_col) / len(entropy_trained_col), 2)
    results_table.add_row(results_table.field_names)
    results_table.add_row(["Average", d_correct_untrained_avg, d_correct_trained_avg, improvement_avg,
                           entropy_untrained_avg, entropy_trained_avg, "-----------"])
    print(results_table)


# Some calculations to explain entropy values:
signal_36_bins_uniform = [i for i in range(36)]
signal_24_bins_uniform = [i for i in range(24)]
signal_12_bins_uniform = [i for i in range(12)]
signal_6_bins_uniform = [i for i in range(6)]

print("\nTypical entropy values:")
print(f"Entropy of a trajectory with 36 states (uniform): {round(calculate_entropy(signal_36_bins_uniform), 2)}")
print(f"Entropy of a trajectory with 24 states (uniform): {round(calculate_entropy(signal_24_bins_uniform), 2)}")
print(f"Entropy of a trajectory with 12 states (uniform): {round(calculate_entropy(signal_12_bins_uniform), 2)}")
print(f"Entropy of a trajectory with 6 states (uniform): {round(calculate_entropy(signal_6_bins_uniform), 2)}")