import numpy as np


def generate_circle_of_fifth_distances():
    """
    distances from the note 0
    :return: a dictionary of notes MIDI notes (modulo 12) with their distance from the initial (0)
    """
    dist_dict = {0:0}
    curr_dist = 0
    curr_notes = [0]
    next_notes = []
    while len(dist_dict) < 12:
        curr_dist += 1
        for note in curr_notes:
            if (((note - 5) % 12) not in next_notes):
                next_notes += [((note - 5) % 12)]
            if (((note + 5) % 12) not in next_notes):
                next_notes += [((note + 5) % 12)]
            if (((note - 7) % 12) not in next_notes):
                next_notes += [((note - 7) % 12)]
            if (((note + 7) % 12) not in next_notes):
                next_notes += [((note + 7) % 12)]
        curr_notes = [item for item in next_notes]
        for note in curr_notes:
            if note not in dist_dict.keys():
                dist_dict[note] = curr_dist
        next_notes = []
    return dist_dict


def scalar_COF_metric(x, y):
    """
    distance according to the output of generate_circle_of_fifth_distances()
    :param x: scalar 1 (int or float)
    :param y: scalar 2 (int or float)
    :return: the distance between x and y
    """
    a = x % 12
    b = y % 12
    dist_dict = generate_circle_of_fifth_distances()
    key = abs(a - b)
    return dist_dict[key]

def scalar_harmonic_metric(x, y):
    """
    distance according to a metric that seems fine for me (Omri) might of course not be good enough...
    :param x: scalar 1 (int or float)
    :param y: scalar 2 (int or float)
    :return: the distance between x and y
    """
    a = x % 12
    b = y % 12
    dist_dict = {0:0, 1:15, 2:10, 3:3, 4:2, 5:1, 6:20, 7:1, 8:2, 9:3, 10:10, 11:15}
    key = abs(a - b)
    return dist_dict[key]

def general_scalar_modulo_12_metric(x, y, dist_dict={0:0, 1:15, 2:10, 3:3, 4:2, 5:1, 6:20, 7:1, 8:2, 9:3, 10:10, 11:15}):
    """
    distance according to a general metric for notes after modulo 12 (distances provided in dictionary)
    :param x: scalar 1 (int or float)
    :param y: scalar 2 (int or float)
    :param dist_dict: a dictionary with the "musical distances" fitting the numerical distances
    :return: the distance between x and y
    """
    dict_keys = sorted(d.keys)
    if dict_keys != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        raise ValueError("general_modulo_12_metric called with dict.keys() != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]")
    a = x % 12
    b = y % 12
    key = abs(a - b)
    return dist_dict[key]

def general_vector_modulo_12_metric(x, y, dist_dict={0:0, 1:15, 2:10, 3:3, 4:2, 5:1, 6:20, 7:1, 8:2, 9:3, 10:10, 11:15}):
    """
    calculating the distance between 2 vectors, using element-wise: general_scalar_modulo_12_metric() func
    :param x: vector 1 (list or np.ndarray)
    :param y: vector 2 (list or np.ndarray)
    :param dist_dict: a dictionary with the "musical distances" fitting the numerical distances
    :return: the distance between x and y
    """
    dict_keys = sorted(dist_dict.keys())
    if dict_keys != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        raise ValueError("general_modulo_12_metric called with dict.keys() != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]")
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError("In func general_vector_modulo_12_metric, var x is of unsupported type")
    if not isinstance(y, (list, np.ndarray)):
        raise TypeError("In func general_vector_modulo_12_metric, var x is of unsupported type")
    if isinstance(x, np.ndarray):
        if x.ndim != 1:
            raise IndexError("In func general_vector_modulo_12_metric, arrays must be of dim=1, given dim=" + str(x.ndim))
    if isinstance(y, np.ndarray):
        if y.ndim != 1:
            raise IndexError("In func general_vector_modulo_12_metric, arrays must be of dim=1, given dim=" + str(x.ndim))
    if isinstance(x, list):
        a = np.array(x)
    else:
        a = x
    if isinstance(y, list):
        b = np.array(y)
    else:
        b = y
    if a.shape[0] != b.shape[0]:
        raise IndexError("In func general_vector_modulo_12_metric, shape[0] of x and y must be equal")
    a = a % 12
    b = b % 12
    c = a - b
    c = np.abs(c)
    distances = [dist_dict[item] for item in c]
    return sum(distances)