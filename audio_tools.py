import numpy as np


def pseudo_scale_estimation(vectorized_midi):
    """
    returns data that may allow to decide which notes are relevant and which aren't
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :return: [dict with all observed notes and number of times, list of observed notes, ...
              ... notes to avoid, validity of notes to avoid (validity is 0 iff there are 3 consecutive semi-tones)]
    """
    notes_in_octave = 12
    counting_dict = {}
    observed_notes = []
    notes_to_avoid = []
    is_valid_notes_to_avoid = 1
    for i in range(notes_in_octave):
        counting_dict[i] = 0
    sum_array_axis_1 = np.sum(vectorized_midi, axis=1)
    for i in range(sum_array_axis_1.shape[0]):
        key = i % 12
        counting_dict[key] += sum_array_axis_1[i]
    for key in counting_dict.keys():
        if counting_dict[key] > 0:
            observed_notes += [key]
    observed_notes = sorted(observed_notes)
    for note in observed_notes:
        if ((note + 1) % 12) in observed_notes:
            if (((note - 1) % 12) not in notes_to_avoid):
                notes_to_avoid += [((note - 1) % 12)]
            if (((note + 2) % 12) not in notes_to_avoid):
                notes_to_avoid += [((note + 2) % 12)]
        if ((note + 2) % 12) in observed_notes and (((note + 1) % 12) not in notes_to_avoid):
            notes_to_avoid += [((note + 1) % 12)]
        if (((note + 1) % 12) in observed_notes) and (((note + 2) % 12) in observed_notes):
            is_valid_notes_to_avoid = 0
    return [counting_dict, observed_notes, notes_to_avoid, is_valid_notes_to_avoid]




def estimate_scale(notes_array):
    """
    :param notes_array: np array with 1 dimension. Representing the song after modulo encoding.
    :return: estimation for the scale, and confidence level between 1 and 5
    """
    print("DON'T USE THIS FUNCTION")
    observed_notes_dict = {}
    for note in notes_array:
        if note in observed_notes_dict.keys():
            observed_notes_dict[note] += 1
        else:
            observed_notes_dict[note] = 1

    observed_notes_list = list(observed_notes_dict.keys())
    notes_to_avoid = []

    # Notes to avoid:
    for i in range(len(observed_notes_dict)):
        for j in range(i, len(observed_notes_dict), 1):
            if (observed_notes_list[i] - observed_notes_list[j]) == 1:
                None
