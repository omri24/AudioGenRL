import numpy as np
import mido
import math

def align_MIDI_timing(midi_object):
    """
    align the timing of notes in the file to sixteenth
    :param midi_object: any MIDI object (from mido library)
    :return: the MIDI file with aligned timings
    """
    ticks_per_sixteenth = midi_object.ticks_per_beat / 4   # assuming the song is 4/4
    for i, track in enumerate(midi_object.tracks):
        for message in track:
            if message.type in ["note_on", "note_off"]:
                initial_timing = message.time
                timing_offset = initial_timing % ticks_per_sixteenth
                if timing_offset != 0:
                    if timing_offset > 0.5 * ticks_per_sixteenth:
                        message.time += ticks_per_sixteenth
                        message.time += (-1) * timing_offset
                    if timing_offset < 0.5 * ticks_per_sixteenth:
                        message.time += (-1) * timing_offset
    return midi_object

def vectorize_MIDI(file_name, num_of_MIDI_notes=128, channel_filtering=-1):
    """
    Assuming only 4/4 songs in the MIDI file and that resolution of 1/16 is small enough
    :param file_name: the midi file to process
    :param num_of_MIDI_notes: number of MIDI notes in the format (usually 128)
    :return: list of numpy arrays, each represent a track from the MIDI file in 1 hot format...
                                    ...(more than 1 note in each slot is possible)
    """
    ret_lst = []
    to_filter_channels = False
    if channel_filtering >= 0:
        selected_channel = channel_filtering
        to_filter_channels = True
    midi_object = mido.MidiFile(file_name)
    midi_object = align_MIDI_timing(midi_object)
    ticks_per_sixteenth = midi_object.ticks_per_beat / 4   # assuming the song is 4/4
    for i, track in enumerate(midi_object.tracks):
        track_array = -1
        note_messages = []
        for message in track:
            if message.type in ["note_on", "note_off"]:
                if (not to_filter_channels) or (to_filter_channels and (message.channel == selected_channel)):
                    note_messages += [message]
        if len(note_messages) == 0:
            print("In current input file, track " + str(i) + " doesn't contain notes")
        else:
            list_for_array = [0 for i in range(num_of_MIDI_notes)]
            for message in note_messages:
                if message.time == 0:
                    if message.type == "note_on":
                        list_for_array[message.note] = 1
                    if message.type == "note_off":
                        list_for_array[message.note] = 0
                else:
                    block_length = message.time / ticks_per_sixteenth  # block_length = round(message.time / ticks_per_sixteenth)     # math.ceil(message.time / ticks_per_sixteenth)
                    vec_for_array = np.array(list_for_array)
                    vec_for_array = np.reshape(vec_for_array, (num_of_MIDI_notes, 1))
                    block_to_concat = np.concatenate([vec_for_array for i in range(int(block_length))], axis=1)
                    if isinstance(track_array, int):
                        track_array = block_to_concat
                    else:
                        track_array = np.concatenate([track_array, block_to_concat], axis=1)
                    if message.type == "note_on":
                        list_for_array[message.note] = 1
                    if message.type == "note_off":
                        list_for_array[message.note] = 0
        if isinstance(track_array, np.ndarray):
            ret_lst += [track_array]
    return ret_lst


def export_MIDI(list_of_track_arrays, file_name="output.mid", ticks_per_sixteenth=120):
    """
    gets a list of arrays, each array represents a "1-hot" MIDI track, and writes the track to MIDI file
    :param list_of_track_arrays: list of "1-hot" arrays (like the output of vectorize_MIDI function)
    :param ticks_per_sixteenth: see the definition in vectorize_MIDI function
    :return: None
    """
    mid = mido.MidiFile()
    for array in list_of_track_arrays:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        sixteenths_from_last_event = -1
        on_going_notes = []
        notes_to_start = []
        notes_to_end = []
        for col in range(array.shape[1]):
            sixteenths_from_last_event += 1
            for row in range(array.shape[0]):
                if (row in on_going_notes) and (array[row, col] == 0):
                    notes_to_end += [row]
                if (row not in on_going_notes) and (array[row, col] == 1):
                    notes_to_start += [row]
            if len(notes_to_start) > 0:
                for note in notes_to_start:
                    on_going_notes += [note]
                    track.append(mido.Message("note_on", note=note, velocity=113,
                                              time=sixteenths_from_last_event * ticks_per_sixteenth))
                    sixteenths_from_last_event = 0
                notes_to_start = []
            if len(notes_to_end) > 0:
                for note in notes_to_end:
                    index_to_remove = on_going_notes.index(note)
                    temp_var = on_going_notes.pop(index_to_remove)
                    track.append(mido.Message("note_off", note=note, velocity=113,
                                              time=sixteenths_from_last_event * ticks_per_sixteenth))
                    sixteenths_from_last_event = 0
                notes_to_end = []
    mid.save(file_name)
    print("MIDI file exported")
    return None
