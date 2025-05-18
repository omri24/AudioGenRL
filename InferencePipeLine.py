import random
import re
import time
import MIDI_IO as io
import MIDI_coding as code
import RL_algorithms as RL
from RL_algorithms import FiniteHorizonDDPEnv, get_top_actions, calculate_entropy
from stable_baselines3 import PPO
import mido
from mido import Message
from copy import deepcopy
from music21 import converter, key, note, stream, analysis
from audio_tools import get_scale_notes, closest_note, remove_risky_notes, move_note_to_correct_octave, handle_semi_tones

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
steps_in_the_final_generation = 2000
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
print("Generating trajectory by PPO")
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
#n = io.export_MIDI([decoded_generated_tuple], file_name="mid_saved_midi_file.mid", ticks_per_sixteenth=180)
timer_end = time.time()
print(f"Trajectory generated in {round(timer_end - timer_start, 2)} seconds by PPO")


# Audio generation params
manual_offset = 0  # Semi-tone shifts up or down to the entire generated audio
target_channel = 3    # Set this to the channel of the lead instrument
file_name_for_inference = "solo and back.MID"
file_name_back = "back.mid"
p = 1 / 2   # Probability to change a note (error)
error_range = 2   # Range of error
squeeze_notes_to_same_octave = False
move_notes_to_input_octave = True

consider_user_playing_direction = True
use_generated_tuple_for_correction = True

auto_set_manual_offset = False

force_key = False
remove_risky_notes_from_key = False

consider_distance_from_back = False
relevant_channels_for_currently_playing = [11]    # Set this to the main instrument that present in the back

# Define a string that will be included in the (fixed) output file name
fix_config_str = "-configs"
if use_generated_tuple_for_correction:
    fix_config_str += "-gen1"
else:
    fix_config_str += "-gen0"
if move_notes_to_input_octave:
    fix_config_str += "-movToIn1"
else:
    fix_config_str += "-movToIn0"
if consider_user_playing_direction:
    fix_config_str += "-direction1"
else:
    fix_config_str += "-direction0"
if force_key:
    fix_config_str += "-forceKey1"
else:
    fix_config_str += "-forceKey0"
if remove_risky_notes_from_key:
    fix_config_str += "-delRisk1"
else:
    fix_config_str += "-delRisk0"
if consider_distance_from_back:
    fix_config_str += "-distBack1"
else:
    fix_config_str += "-distBack0"
if squeeze_notes_to_same_octave:
    fix_config_str += "-squeeze1"
else:
    fix_config_str += "-squeeze0"
if auto_set_manual_offset:
    fix_config_str += "-autoManualOffset1"
else:
    fix_config_str += "-autoManualOffset0"

# If 'force_key' and 'consider_distance_from_back', use 'consider_distance_from_back'
if force_key and consider_distance_from_back:
    force_key = False
    print("!!! Can't use both 'force_key' and 'consider_distance_from_back' - 'force_key' disabled !!!")

# Inference
timer_start = time.time()
print("Starting inference run")
mid = mido.MidiFile(file_name_for_inference)

# Create errors
mid_errors = deepcopy(mid)
for i, track in enumerate(mid.tracks):
    for j, msg in enumerate(track):
        if msg.type == "note_on" and msg.channel == target_channel:
            ending_message = Message('note_off', channel=msg.channel, note=msg.note,
                                          velocity=msg.velocity, time=msg.time)
            curr_end_idx = "not found"
            for k, _msg in enumerate(track):
                if _msg.type == "note_off":
                    #ending_message.time = _msg.time
                    #ending_message.velocity = _msg.velocity

                    if k > j and _msg.channel == target_channel and _msg.note == msg.note:
                        curr_end_idx = k
                        break
            if isinstance(curr_end_idx, str):
                print(f"During error making found a note without 'note_off' message - will not be changed")
                continue
            to_cause_error = random.randint(0, int(1 / p) - 1)
            if to_cause_error == 0:
                to_cause_error = 1
            else:
                to_cause_error = 0
            error_size = random.randint(1, error_range) * to_cause_error
            error_sign = random.randint(0, 1)
            if error_sign == 0:
                error_sign = -1
            else:
                error_sign = 1
            msg_to_errors = deepcopy(msg)
            #msg_to_errors.note += error_size
            #ending_message.note += error_size
            total_error = error_size * error_sign
            mid_errors.tracks[i][j].note += total_error
            mid_errors.tracks[i][curr_end_idx].note += total_error
mid_errors.save('inference_output_errors.mid')    # File saved just to be opened

mid = mido.MidiFile('inference_output_errors.mid')

# Estimate keys
midi_input_only_back = converter.parse(file_name_back)
midi_input_used_in_training = converter.parse(reference_file)

#midi_generated = converter.parse("mid_saved_midi_file.mid")

key_estimate_input_only_back = midi_input_only_back.analyze('key').name
key_estimate_input_only_back_relative_eq = midi_input_only_back.analyze('key').relative.name

key_estimate_input_used_in_training = midi_input_used_in_training.analyze('key').name
key_estimate_input_used_in_training_relative_eq = midi_input_used_in_training.analyze('key').relative.name

#key_estimate_generated = midi_generated.analyze('key').name

key_notes_input_only_back = get_scale_notes(key_estimate_input_only_back)
if remove_risky_notes_from_key:
    key_notes_input_only_back = remove_risky_notes(key_notes_input_only_back)

# If selected - set the manual_offset automatically to eliminate scale difference
if auto_set_manual_offset:
    lst_options_for_training_scale = [key_estimate_input_used_in_training, key_estimate_input_used_in_training_relative_eq]
    lst_options_for_user_backing_track = [key_estimate_input_only_back, key_estimate_input_only_back_relative_eq]
    for item in lst_options_for_training_scale:
        if re.search("major", item):
            training_scale_major = item
        else:
            training_scale_minor = item
    for item in lst_options_for_user_backing_track:
        if re.search("major", item):
            backing_track_scale_major = item
        else:
            backing_track_scale_minor = item
    note_map = {"C": 0, "C#": 1, "D-": 1, "D": 2,
                "E-": 3, "E": 4, "F": 5, "F#": 6,
                "G-": 6, "G": 7, "A-": 8, "A": 9,
                "B-": 10, "B": 11, "C-": 11}

    training_scale_note_name = training_scale_major[:training_scale_major.index(" ")]
    backing_track_note_name = backing_track_scale_major[:backing_track_scale_major.index(" ")]

    distance_back_train = note_map[backing_track_note_name] - note_map[training_scale_note_name]
    if distance_back_train > 6:   # Case need to reduce instead of increase
        distance_back_train += -12

    manual_offset = distance_back_train

# Remove unnecessary items from tuple (666) and make sure the distances are less than an octave
generated_tuple_lst = list(generated_tuple)
for idx, item in enumerate(generated_tuple_lst):
    if item > 127:   # Remove 666 and 999
        if idx > 0:
            generated_tuple_lst[idx] = generated_tuple_lst[idx - 1]
        else:   # idx == 0
            generated_tuple_lst[idx] = generated_tuple[0]
    if idx > 0 and squeeze_notes_to_same_octave:
        last_note = generated_tuple_lst[idx -1]
        if abs(last_note - item) > 11:
            if item > last_note:
                generated_tuple_lst[idx] += -12
            else:  # last_note >= item
                generated_tuple_lst[idx] += 12

generated_tuple = tuple(generated_tuple_lst)

# Fix generated tuple
mid_fixed = deepcopy(mid)
pointer_on_generated_tuple = 0

for i, track in enumerate(mid.tracks):

    currently_playing = []
    backup_memory = []
    last_note_from_user = -1
    last_selected_note = -1

    # Backup memory must not be empty - fill it with some relevant note
    for temp_msg in enumerate(track):
        if temp_msg == "note_on" and msg.channel in relevant_channels_for_currently_playing:
            backup_memory.append(msg.note)

    # Iterate over all messages
    for j, msg in enumerate(track):
        if msg.type == "note_on" and msg.channel == target_channel:
            if msg.velocity > 0:
                ending_message = Message('note_off', channel=msg.channel, note=msg.note,
                                              velocity=msg.velocity, time=msg.time)
                curr_end_idx = "not found"
                for k, _msg in enumerate(track):
                    if _msg.type == "note_off":
                        #ending_message.time = _msg.time
                        #ending_message.velocity = _msg.velocity

                        if k > j and _msg.channel == target_channel and _msg.note == msg.note:
                            curr_end_idx = k
                            break
                if isinstance(curr_end_idx, str):
                    print(f"During error correction found a note without 'note_off' message - will not be changed")
                    continue
                if use_generated_tuple_for_correction:
                    note_to_use =  generated_tuple[pointer_on_generated_tuple]
                else:
                    note_to_use = msg.note


                # Handle direction
                if consider_user_playing_direction:
                    if msg.note > last_note_from_user:
                        direction = 1
                    elif msg.note < last_note_from_user:
                        direction = -1
                    else:  # msg.note == last_note_from_user
                        direction = 0

                    # Find offset
                    note_to_use_mod_12 = note_to_use % 12
                    if note_to_use_mod_12 not in key_notes_input_only_back:
                        fixed_note_mod_12 = closest_note(note_to_use_mod_12, key_notes_input_only_back)
                        if not remove_risky_notes_from_key:
                            fixed_note_mod_12 = handle_semi_tones(fixed_note_mod_12, [i[1] for i in currently_playing])
                        scale_offset = fixed_note_mod_12 - note_to_use_mod_12
                    else:  # Note in scale
                        scale_offset = 0
                    if note_to_use_mod_12 not in backup_memory:
                        fixed_note_mod_12 = closest_note(note_to_use_mod_12, [i[1] for i in backup_memory])
                        back_offset = fixed_note_mod_12 - note_to_use_mod_12
                    else:  # Note in back
                        back_offset = 0
                    calc_offset = manual_offset + scale_offset * int(force_key) + back_offset * int(consider_distance_from_back)

                    if direction == 1 and calc_offset + note_to_use < last_selected_note:  # User went up but generated audio will go down
                        note_to_use += 12
                    elif direction == -1 and calc_offset + note_to_use > last_selected_note:  # User went down but generated audio will go ip
                        note_to_use += -12
                    else:  # direction == 0
                        note_to_use = last_selected_note
                else:
                    direction = 1  # to not affect the offset

                if move_notes_to_input_octave:
                    note_to_use = move_note_to_correct_octave(note_to_use, msg.note)   # Move to correct octave
                else:
                    print("Inference without 'move_notes_to_input_octave' set to True (as currently executed) is not safe!")

                note_to_use_mod_12 = note_to_use % 12
                if note_to_use_mod_12 not in key_notes_input_only_back:
                    fixed_note_mod_12 = closest_note(note_to_use_mod_12, key_notes_input_only_back)
                    if not remove_risky_notes_from_key:
                        fixed_note_mod_12 = handle_semi_tones(fixed_note_mod_12, [i[1] for i in currently_playing])
                    scale_offset = fixed_note_mod_12 - note_to_use_mod_12
                else:  # Note in scale
                    scale_offset = 0
                if note_to_use_mod_12 not in backup_memory:
                    fixed_note_mod_12 = closest_note(note_to_use_mod_12, [i[1] for i in backup_memory])
                    back_offset = fixed_note_mod_12 - note_to_use_mod_12
                else:  # Note in back
                    back_offset = 0

                if not consider_user_playing_direction or (consider_user_playing_direction and direction != 0):
                    pointer_on_generated_tuple += 1
                msg_to_fixed = deepcopy(msg)
                #msg_to_fixed.note += scale_offset * int(force_key)
                #ending_message.note += scale_offset * int(force_key)
                total_offset = (manual_offset + scale_offset * int(force_key) + back_offset * int(consider_distance_from_back)) * abs(direction)
                mid_fixed.tracks[i][j].note = note_to_use + total_offset
                mid_fixed.tracks[i][curr_end_idx].note = note_to_use + total_offset
                last_note_from_user = msg.note
                last_selected_note = note_to_use + total_offset
                #key = (target_channel, mid_fixed.tracks[i][j].note % 12)
                #if key not in currently_playing:
                #    currently_playing.append(key)

        # Not in target channel, but it's note_on/note_off, also, avoid drums
        elif msg.type in ['note_on', 'note_off'] and msg.channel in relevant_channels_for_currently_playing:
            key = (msg.channel, msg.note % 12)
            if msg.type == 'note_on' and msg.velocity > 0:
                if key not in currently_playing:
                    currently_playing.append(key)
            else:  # note_off or note_on with velocity 0
                if key in currently_playing:
                    currently_playing.remove(key)
            if len(currently_playing) > 0:   # There are notes to relate to - update the memory
                backup_memory = deepcopy(currently_playing)

mid_fixed.save(f'inference_output_fixed{fix_config_str}.mid')
timer_end = time.time()
print(f"Inference succeed files saved, in {round(timer_end - timer_start, 2)} seconds")




