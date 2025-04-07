import numpy as np
from copy import deepcopy

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



NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11
}

# Interval patterns (in semitones) for each scale type
SCALE_INTERVALS = {
    "major":        [0, 2, 4, 5, 7, 9, 11],
    "minor":        [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic minor":  [0, 2, 3, 5, 7, 9, 11],
    "dorian":       [0, 2, 3, 5, 7, 9, 10],
    "phrygian":     [0, 1, 3, 5, 7, 8, 10],
    "lydian":       [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":   [0, 2, 4, 5, 7, 9, 10],
    "locrian":      [0, 1, 3, 5, 6, 8, 10],
    "blues":        [0, 3, 5, 6, 7, 10],
    "pentatonic major": [0, 2, 4, 7, 9],
    "pentatonic minor": [0, 3, 5, 7, 10]
}

def get_scale_notes(scale_name):
    try:
        parts = scale_name.strip().lower().split()
        if len(parts) < 2:
            raise ValueError("Invalid scale format. Use format like 'A major'.")

        root_note = parts[0].capitalize()
        scale_type = ' '.join(parts[1:]).lower()

        if root_note not in NOTE_TO_MIDI:
            raise ValueError(f"Unknown root note: {root_note}")
        if scale_type not in SCALE_INTERVALS:
            raise ValueError(f"Unknown scale type: {scale_type}")

        root_midi = NOTE_TO_MIDI[root_note]
        intervals = SCALE_INTERVALS[scale_type]

        return [(root_midi + i) % 12 for i in intervals]

    except Exception as e:
        print(f"Error: {e}")
        return []


def closest_note(original_note, legal_notes):
    return min(legal_notes, key=lambda x: (abs(x - original_note), legal_notes.index(x)))

def remove_risky_notes(notes_lst):
    lst_copy = deepcopy(notes_lst)
    idx_to_remove = []
    for i, item in enumerate(lst_copy):
        if item + 1 in lst_copy or item - 1 in lst_copy:
            idx_to_remove.append(i)
    idx_to_remove.sort(reverse=True)
    for idx in idx_to_remove:
        temp_var = lst_copy.pop(idx)
    return lst_copy


def move_note_to_correct_octave(src, target):
    closest = src
    min_distance = abs(src - target)

    # Try shifting src up or down by octaves (-4 to +4 as a reasonable range)
    for octave_shift in range(-4, 5):
        shifted = src + octave_shift * 12
        distance = abs(shifted - target)
        if distance < min_distance:
            min_distance = distance
            closest = shifted

    return closest