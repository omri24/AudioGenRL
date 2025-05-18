import os
import random
import mido
import time
import pandas as pd
import MIDI_IO as io
import MIDI_coding as code
import RL_algorithms as RL
from RL_algorithms import FiniteHorizonDDPEnv, get_top_actions, calculate_entropy
import audio_metrics as metrics
from stable_baselines3 import PPO
from prettytable import PrettyTable
import re
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


# Configurations
reference_file = "piano1_ref.mid"
file_to_fix = "single_notes_errors_piano_and_drums6.mid"
correct_file = "single_notes_piano_and_drums6.mid"
path_to_snap_shots = "ppo_checkpoints"
ground_truths_lst = [("solo and back.MID", 0, 3),
                     ("MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi", 1, 0),
                     ("MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--4.midi", 1, 0)]   # Elements must be: (file_name, target_track, target_channel)

if any(char.isdigit() for char in path_to_snap_shots):
    print("Warning! path_to_snap_shots contains numbers - that will create wrong time step vals in the tales")


# Hyper-parameters
len_of_state = 4
top_n = 10   # Number of items to take after for generating the final policy
top_n_untrained = top_n
arcs_for_state = 10   # Minimal number of actions that will be legal for each state
horizon = 500
steps_in_the_final_generation = 600
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

ppo_zips = os.listdir(path_to_snap_shots)
ppo_zips = [os.path.join(path_to_snap_shots, item) for item in ppo_zips]

# Define the result table
results_table = PrettyTable(["Time steps", "MIDI file", "Mean delta[correct, untrained]", "Mean delta[correct, trained]",
                             "Improvement (%)", "Entropy untrained", "Entropy trained", "Wasserstein distance untrained",
                             "Wasserstein distance trained", "Reward untrained", "Reward trained",
                             "Time untrained [s]", "Time trained [s]"])
# Load the untrained model
untrained_model = PPO.load("ppo_checkpoints/model_0_steps.zip")
for zip_file in ppo_zips:

    # Result table per time step
    temporal_results_table = PrettyTable(
        ["Time steps", "MIDI file", "Mean delta[correct, untrained]", "Mean delta[correct, trained]",
         "Improvement (%)", "Entropy untrained", "Entropy trained", "Wasserstein distance untrained",
                             "Wasserstein distance trained", "Reward untrained", "Reward trained",
                            "Time untrained [s]", "Time trained [s]"])

    # Collect time steps
    curr_time_steps = re.sub(r'\D', '', zip_file)

    # Load the trained model
    model = PPO.load(zip_file)

    # Extract the policy learned by the agent (PPO)
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

    # Get the rewards dict
    rewards_dict_vec_states = agent_tools.env.all_rewards

    # Generate audio with trained and untrained policies
    timer_start = time.time()
    generated_tuple = ()
    reward_trained = 0
    idx = random.randint(0, len(list(vec_policy_ppo.keys())) - 1)
    s = list(vec_policy_ppo.keys())[idx]
    for _ in range(steps_in_the_final_generation):
        options_lst = vec_policy_ppo[s]
        idx = random.randint(leading_items_to_remove_from_action_options, len(options_lst) - 1)
        s_prev = s
        s = vec_policy_ppo[s][idx]
        curr_reward = rewards_dict_vec_states.get((s_prev, s), 0)
        reward_trained += curr_reward
        generated_tuple += s


        decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple)
        #n = io.export_MIDI([decoded_generated_tuple], file_name="out_new.mid", ticks_per_sixteenth=180)
    timer_end = time.time()
    calc_time_trained = timer_end - timer_start

    # Generate with untrained model
    timer_start = time.time()
    generated_tuple_untrained = ()
    reward_untrained = 0
    idx = random.randint(0, len(list(vec_policy_ppo_untrained.keys())) - 1)
    s = list(vec_policy_ppo_untrained.keys())[idx]
    for _ in range(steps_in_the_final_generation):
        options_lst = vec_policy_ppo_untrained[s]
        idx = random.randint(0, len(options_lst) - 1)
        s_prev = s
        s = vec_policy_ppo_untrained[s][idx]
        curr_reward = rewards_dict_vec_states.get((s_prev, s), 0)
        reward_untrained += curr_reward
        generated_tuple_untrained += s
    timer_end = time.time()
    calc_time_untrained = timer_end - timer_start

    def extract_note_pitches(midi_path, target_track, target_channel):
        note_pitches = []

        # Load the MIDI file
        try:
            mid = mido.MidiFile(midi_path)
        except IOError:
            print(f"Could not open MIDI file: {midi_path}")
            return []

        # Iterate through all messages in all tracks
        for track_idx, track in enumerate(mid.tracks):
            if track_idx == target_track:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        if msg.channel == target_channel:
                            note_pitches.append(msg.note)

        return tuple(note_pitches)

    # Rename tuple
    generated_tuple_trained = generated_tuple

    # Remove non-midi items (used in training but irrelevant now)
    def replace_non_midi(input_tuple):
        # Convert to list since tuples are immutable
        input_list = list(input_tuple)

        # Identify indices and values of elements
        small_values = [val for val in input_tuple if val < 128]
        if not small_values:
            raise ValueError("No values < 128 found to replace large values.")

        # Replace elements > 127 with random choice from small_values
        for i, val in enumerate(input_list):
            if val > 127:
                input_list[i] = random.choice(small_values)

        return tuple(input_list)

    generated_tuple_untrained = replace_non_midi(generated_tuple_untrained)
    generated_tuple_trained = replace_non_midi(generated_tuple_trained)

    # Get ground truth tuples
    ground_truth_tuples_last = [extract_note_pitches(*item) for item in ground_truths_lst]

    # Calculate the improvement
    for audio_idx, correct_tuple in enumerate(ground_truth_tuples_last):
        shortest_sequence_len = min(len(correct_tuple), len(generated_tuple_trained), len(generated_tuple_untrained))
        # other dict option dist_dict={0: 0, 1: 5, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6, 7: 1, 8: 4, 9: 3, 10: 2, 11: 5}
        delta_correct_untrained = metrics.general_vector_modulo_12_metric(
            list(correct_tuple)[:shortest_sequence_len], list(generated_tuple_untrained)[:shortest_sequence_len])
        delta_correct_trained = metrics.general_vector_modulo_12_metric(
            list(correct_tuple)[:shortest_sequence_len], list(generated_tuple_trained)[:shortest_sequence_len])
        mean_delta_untrained = delta_correct_untrained / shortest_sequence_len
        mean_delta_trained = delta_correct_trained / shortest_sequence_len

        # Positive value of 'improvement' is what we want
        improvement = (mean_delta_untrained - mean_delta_trained) / mean_delta_untrained

        # Check the entropy of the generated audio
        entropy_untrained = calculate_entropy(generated_tuple_untrained)
        entropy_trained = calculate_entropy(generated_tuple_trained)
        entropy_correct = calculate_entropy(correct_tuple)

        # Find Wasserstein distances
        wasserstein_distance_untrained= wasserstein_distance(correct_tuple, generated_tuple_untrained)
        wasserstein_distance_trained= wasserstein_distance(correct_tuple, generated_tuple_trained)

        curr_file_name = ground_truths_lst[audio_idx][0]
        results_table.add_row([curr_time_steps, curr_file_name,
                               round(mean_delta_untrained, 2), round(mean_delta_trained, 2),
                              round(improvement * 100, 2), round(entropy_untrained, 2), round(entropy_trained, 2),
                               round(wasserstein_distance_untrained, 2), round(wasserstein_distance_trained, 2),
                               round(reward_untrained, 2), round(reward_trained, 2),
                               round(calc_time_untrained, 10), round(calc_time_trained, 10)])

        temporal_results_table.add_row([curr_time_steps, curr_file_name,
                                        round(mean_delta_untrained, 2), round(mean_delta_trained, 2),
                              round(improvement * 100, 2), round(entropy_untrained, 2), round(entropy_trained, 2),
                                        round(wasserstein_distance_untrained, 2), round(wasserstein_distance_trained, 2),
                                        round(reward_untrained, 2), round(reward_trained, 2),
                                        round(calc_time_untrained, 10), round(calc_time_trained, 10)])
    # Find averages
    mean_d_correct_untrained = [float(row[2]) for row in temporal_results_table._rows]
    mean_d_correct_trained = [float(row[3]) for row in temporal_results_table._rows]
    improvement_col = [float(row[4]) for row in temporal_results_table._rows]
    entropy_untrained_col = [float(row[5]) for row in temporal_results_table._rows]
    entropy_trained_col = [float(row[6]) for row in temporal_results_table._rows]
    wasserstein_untrained_col = [float(row[7]) for row in temporal_results_table._rows]
    wasserstein_trained_col = [float(row[8]) for row in temporal_results_table._rows]
    reward_untrained = [float(row[9]) for row in temporal_results_table._rows]
    reward_trained = [float(row[10]) for row in temporal_results_table._rows]
    time_untrained_col = [float(row[11]) for row in temporal_results_table._rows]
    time_trained_col = [float(row[12]) for row in temporal_results_table._rows]

    d_correct_untrained_avg = round(sum(mean_d_correct_untrained) / len(mean_d_correct_untrained), 2)
    d_correct_trained_avg = round(sum(mean_d_correct_trained) / len(mean_d_correct_trained), 2)
    improvement_avg = round(sum(improvement_col) / len(improvement_col), 2)
    entropy_untrained_avg = round(sum(entropy_untrained_col) / len(entropy_untrained_col), 2)
    entropy_trained_avg = round(sum(entropy_trained_col) / len(entropy_trained_col), 2)
    wasserstein_untrained_avg = round(sum(wasserstein_untrained_col) / len(wasserstein_untrained_col), 2)
    wasserstein_trained_avg = round(sum(wasserstein_trained_col) / len(wasserstein_trained_col), 2)
    reward_untrained_avg = round(sum(reward_untrained) / len(reward_untrained), 2)
    reward_trained_avg = round(sum(reward_trained) / len(reward_trained), 2)
    time_untrained_avg = round(sum(time_untrained_col) / len(time_untrained_col), 10)
    time_trained_avg = round(sum(time_trained_col) / len(time_trained_col), 10)

    temporal_results_table.add_row(results_table.field_names)
    temporal_results_table.add_row([curr_time_steps, "Average", d_correct_untrained_avg, d_correct_trained_avg, improvement_avg,
                           entropy_untrained_avg, entropy_trained_avg, wasserstein_untrained_avg, wasserstein_trained_avg,
                                    reward_untrained_avg, reward_trained_avg, time_untrained_avg, time_trained_avg])
    results_table.add_row(
        [curr_time_steps, "Average", d_correct_untrained_avg, d_correct_trained_avg, improvement_avg,
         entropy_untrained_avg, entropy_trained_avg, wasserstein_untrained_avg, wasserstein_trained_avg,
         reward_untrained_avg, reward_trained_avg, time_untrained_avg, time_trained_avg])
    print(temporal_results_table)


print(results_table)


def prettytable_to_sorted_xlsx(pt, filename="output.xlsx"):
    # Convert PrettyTable to list of rows
    data = [row for row in pt.rows]

    # Extract column names
    columns = pt.field_names

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Ensure leftmost column is numeric for sorting
    leftmost_col = df.columns[0]
    df[leftmost_col] = pd.to_numeric(df[leftmost_col], errors='coerce')

    # Sort and reset index
    df = df.sort_values(by=leftmost_col).reset_index(drop=True)

    # Save to xlsx
    df.to_excel(filename, index=False)
    print(f"Data saved to '{filename}'")

    # Return df
    return df

# Convert to df and sort according to number of time steps
result_df = prettytable_to_sorted_xlsx(results_table, "out_table_complete_analysis.xlsx")
filtered_df = result_df[result_df["MIDI file"].str.contains("Average", na=False)]  # Only averages inside


# Plots
x_axis = filtered_df["Time steps"].tolist()
custom_dist_untrained = filtered_df["Mean delta[correct, untrained]"].tolist()
custom_dist_trained = filtered_df["Mean delta[correct, trained]"].tolist()
im_vals = filtered_df["Improvement (%)"].tolist()
entropy_untrained_vals = filtered_df["Entropy untrained"].tolist()
entropy_trained_vals = filtered_df["Entropy trained"].tolist()
wasserstein_dists_untrained = filtered_df["Wasserstein distance untrained"].tolist()
wasserstein_dists_trained = filtered_df["Wasserstein distance trained"].tolist()
rewards_untrained = filtered_df["Reward untrained"].tolist()
rewards_trained = filtered_df["Reward trained"].tolist()
times_untrained = filtered_df["Time untrained [s]"].tolist()
times_trained = filtered_df["Time trained [s]"].tolist()

print("Average IM: ", sum(im_vals) / len(im_vals))

def plot_two_lines(x, y1, y2, title, labels, x_label, y_label, save_path):
    if len(x) != len(y1) or len(x) != len(y2):
        raise ValueError("All input lists must have the same length.")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=labels[0], marker='o')
    plt.plot(x, y2, label=labels[1], marker='s')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_one_line(x, y, title, label, x_label, y_label, save_path):
    if len(x) != len(y):
        raise ValueError("All input lists must have the same length.")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label, marker='o')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

plot_two_lines(x=x_axis, y1=custom_dist_untrained, y2=custom_dist_trained, title="Harmonic distance to ground truth",
               labels=("Distance untrained", "Distance trained"), x_label="Training time steps",
               y_label="Harmonic steps", save_path="result_plot_harmonic_dist.png")

plot_one_line(x=x_axis, y=im_vals, title="IM values", label="IM values", x_label="Training time steps", y_label="%",
              save_path="result_plot_im_vals.png")

plot_two_lines(x=x_axis, y1=entropy_untrained_vals, y2=entropy_trained_vals, title="Song entropy",
               labels=("Entropy untrained", "Entropy trained"), x_label="Training time steps",
               y_label="Bits", save_path="result_plot_entropy.png")

plot_two_lines(x=x_axis, y1=wasserstein_dists_untrained, y2=wasserstein_dists_trained,
               title="Wasserstein distance to ground truth", labels=("Distance untrained", "Distance trained"),
               x_label="Training time steps", y_label="Distance", save_path="result_plot_wasserstein.png")

plot_two_lines(x=x_axis, y1=times_untrained, y2=times_trained,
               title="Generation time", labels=("Time untrained", "Time trained"),
               x_label="Training time steps", y_label="Time [s]", save_path="result_plot_time.png")

plot_two_lines(x=x_axis, y1=rewards_untrained, y2=rewards_trained,
               title="Rewards", labels=("Rewards untrained", "Rewards trained"), x_label="Training time steps",
               y_label="Reward", save_path="result_plot_rewards.png")

"""
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
"""