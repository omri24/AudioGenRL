
import numpy as np
import time



def single_note_modulo_encoder(single_note_1_hot_arr, modulu=False):
    """
    Encodes a song according to note names, omitting data about the octave. Must 1 or 0 notes at each time slot
    :param single_note_1_hot_arr: output of MIDI_coding.get_single_note_audio_from_multi_note_audio
    :return: np array of the encoding
    """
    sum_array_axis_0 = np.sum(single_note_1_hot_arr, axis=0)
    mask = sum_array_axis_0 > 1
    two_notes_in_one_time_slot = np.any(mask)
    if two_notes_in_one_time_slot:
        raise ValueError("In the midi input there are time slots in which 2 notes are being played - not supported")
    argmax_arr = np.argmax(single_note_1_hot_arr, axis=0)
    mask = argmax_arr == 0
    mask = mask * 666
    if modulu:
        argmax_arr = argmax_arr % 12
    argmax_arr = argmax_arr + mask      # if at time slot t there are no notes, place 666 at this point
    return tuple(argmax_arr)


def vectorized_MIDI_to_modulu_array(vectorized_midi):
    """
    Apply modulo 12 logic (note name - omitting octave data) to the output of "MIDI_IO.vectorize_MIDI"
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :return: np array, after applying modulo 12 logic to each column
    """
    modulo_12_array = np.zeros(shape=(12, vectorized_midi.shape[1]))
    for col in range(vectorized_midi.shape[1]):
        for row in range(vectorized_midi.shape[0]):
            if vectorized_midi[row, col] == 1:
                modulo_row = row % 12
                modulo_12_array[modulo_row, col] = 1
    return modulo_12_array


def format_dataset_single_note_optional_modulo_encoding(vectorized_midi, samples_for_algo=4, modulo_12=0):
    """
    generate dataset, for generative model that generates 1 note for time slot (1/16)
    note: this function formats the MIDI to "feature vectors" and labels that are with the same dimensions
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :param samples_for_algo: number of time slots (1/16) that are passed to the generative algorithm at each generating iteration
    :param modulo_12: if 1, apply modulo 12 to the notes, else, don't apply it
    :return: a dictionary, each key is a tuple with n = look_back entries (curr state), and the value is a dict
             that maps next state and observed amount of occurrence
    """
    start_timer = time.time()
    if modulo_12 == 1:
        array_to_use = vectorized_MIDI_to_modulu_array(vectorized_midi)
    else:
        array_to_use = vectorized_midi
    sum_ax_0 = np.sum(array_to_use, axis=0)
    ret_dict = {}
    for col in range(array_to_use.shape[1]):
        if col + 7 < array_to_use.shape[1]:
            curr_states = []
            next_states = []
            for offset in range(2 * samples_for_algo):
                t_curr_states = []
                t_next_states = []
                if sum_ax_0[col + offset] == 0:    # Handle the situation that no notes are played
                    if (offset > 0) and (offset < samples_for_algo):
                        t_curr_states = [item + [666] for item in curr_states]
                    elif (offset > samples_for_algo) and (offset < (2 * samples_for_algo)):
                        t_next_states = [item + [666] for item in next_states]
                    elif offset == 0:
                        t_curr_states += [[666]]
                    else:   # offset is samples_for_algo
                        t_next_states += [[666]]
                else:
                    for row in range(array_to_use.shape[0]):
                        if array_to_use[row, col + offset] == 1:
                            if offset == 0:
                                t_curr_states += [[row]]
                            elif offset == samples_for_algo:
                                t_next_states += [[row]]
                            elif offset > 0 and offset < samples_for_algo:
                                for item in curr_states:
                                    t_curr_states += [item + [row]]
                            else:    # offset > samples_for_algo and offset < 2 * samples_for_algo
                                for item in next_states:
                                    t_next_states += [item + [row]]
                if offset < samples_for_algo:
                    curr_states = t_curr_states
                else:
                    next_states = t_next_states
            next_states_dict = {}
            for item_list in next_states:
                item_tuple = tuple(item_list)
                if item_tuple not in next_states_dict.keys():
                    next_states_dict[item_tuple] = 1
                else:
                    next_states_dict[item_tuple] += 1
            for list_key in curr_states:
                tuple_key = tuple(list_key)
                if tuple_key not in ret_dict.keys():
                    ret_dict[tuple_key] = next_states_dict
                else:
                    for inner_key in next_states_dict.keys():
                        if inner_key not in ret_dict[tuple_key].keys():
                            ret_dict[tuple_key][inner_key] = next_states_dict[inner_key]
                        else:
                            ret_dict[tuple_key][inner_key] += next_states_dict[inner_key]
    end_timer = time.time()
    calc_time = end_timer - start_timer
    print("Encoder finished after " + str(round(calc_time, 2)) + " seconds")
    return ret_dict


def decode_1d_non_modulo_vectorized_audio(audio_vector_1d):
    """
    takes a 1d sequence of numbers that represent MIDI notes (each in [0, 127]). Returns a fitting 1-hot array.
    :param audio_vector_1d: sequence
    :return:
    """
    arr = np.array(audio_vector_1d)
    vectorized_arr_1_hot = np.zeros(shape=(128, arr.shape[0]))
    for index, item in enumerate(arr):
        if item != 666:
            vectorized_arr_1_hot[item, index] = 1
    return vectorized_arr_1_hot


def get_single_note_audio_from_multi_note_audio(vectorized_midi):
    """
    generates "single note version" from a multi note song - takes only the high note
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :return: np array, 1-hot encoded representing the single note song
    """
    single_note_1_hot_arr = np.zeros(shape=vectorized_midi.shape)
    for col in range(vectorized_midi.shape[1]):
        for row in range(vectorized_midi.shape[0]):
            if vectorized_midi[row, col] != 0:
                single_note_1_hot_arr[row, col] = 1
                break
    return single_note_1_hot_arr

def get_up_down_features_from_audio(single_note_1_hot_arr, len_of_state=False):
    up_down_feature_lst = []
    sum_ax_0_arr = np.sum(single_note_1_hot_arr, axis=0)
    argmax_arr = np.argmax(single_note_1_hot_arr, axis=0)
    last_item = -1
    for idx, item in enumerate(argmax_arr):
        if len_of_state:
            if idx % len_of_state == 0:
                up_down_feature_lst.append(999)  # That says that this is the first note in state
            else:
                if sum_ax_0_arr[idx] != 0:
                    if item > last_item:
                        up_down_feature_lst.append(1)
                    elif item < last_item:
                        up_down_feature_lst.append(-1)
                    else:     # argmax_arr[idx] == last_item
                        up_down_feature_lst.append(0)
                else:     # sum_ax_0_arr[idx] == 0
                    up_down_feature_lst.append(666)
        else:
            if sum_ax_0_arr[idx] != 0:
                if item > last_item:
                    up_down_feature_lst.append(1)
                elif item < last_item:
                    up_down_feature_lst.append(-1)
                else:  # argmax_arr[idx] == last_item
                    up_down_feature_lst.append(0)
            else:  # sum_ax_0_arr[idx] == 0
                up_down_feature_lst.append(666)
        if sum_ax_0_arr[idx] != 0:
            last_item = item
        else:
            last_item = 72    # Choose C5, change if C isn't good for the training data scale
    return up_down_feature_lst
