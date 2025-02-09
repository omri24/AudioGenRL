import numpy as np
import MIDI_IO as io
import MIDI_coding as code
import time


class LinOpt:
    def __init__(self):
        self.features = []
        self.labels = []
        self.repetitions = 0
        self.modulo_param = 0
        self.mode = 0
        self.midi_range = 128
        self.estimation_map = {}   # Converts form real notes space to the estimation space
        self.inverse_estimation_map = {}     # Converts form estimation space to the real notes space

    def construct_basic_mat_from_dict_for_linear_optimal_estimator(self, observations_dict, mode, scalar_labels=True, modulo_param=12):
        """
        :param observations_dict: dict in the format of the output of the function
                                  "MIDI_coding.format_dataset_single_note_optional_modulo_encoding"
                                  see the env constructor in RL algorithms for reference
        :param mode: estimation mode (for example, 0 means estimating scalar from vector)
        """
        timer_start = time.time()
        self.mode = mode
        self.modulo_param = modulo_param
        features_lst = []
        labels_lst = []
        repetitions_lst = []
        for outer_key in observations_dict.keys():
            if 666 in outer_key:
                continue
            feature = np.asarray(outer_key, dtype=np.int32)
            feature = np.reshape(feature, (feature.shape[0], 1))
            for inner_key in observations_dict[outer_key].keys():
                label = np.asarray(inner_key, dtype=np.int32)
                label = np.reshape(label, (label.shape[0], 1))
                repetitions = observations_dict[outer_key][inner_key]
                if scalar_labels:
                    label = inner_key[0]
                if isinstance(label, int):
                    if label == 666:
                        continue
                else:
                    if 666 in label:
                        continue
                repetitions_lst.append(repetitions)
                features_lst.append(feature)
                labels_lst.append(label)
        self.features = np.concatenate(features_lst, axis=1)
        if scalar_labels:
            self.labels = np.asarray(labels_lst, dtype=np.int32)
        else:
            self.labels = np.concatenate(labels_lst, axis=1)
        if modulo_param:
            self.features = self.features % modulo_param
            self.labels = self.labels % modulo_param
        self.repetitions = repetitions_lst
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Vectors constructed in " + str(round(calc_time, 2)) + " seconds")

    def harmonic_dist_for_lin_opt(self, note):
        """
        MIDI code for C always fulfils c % 12 = 0.
        The general map is for the case that c is the mean of the labels
        :param note:
        :return:
        """
        general_map = {0: 0, 1: 5, 2: -2, 3: 3, 4: -4, 5: 1, 6: 6, 7: -1, 8: 4, 9: -3, 10: 2, 11: -5}
        return general_map[note]


    def construct_data_statistics(self):
        timer_start = time.time()
        mode = self.mode
        if mode == 0:   # Estimating scalar x from vector y
            spanned_labels = np.repeat(self.labels, self.repetitions)
            E_x_init = np.mean(spanned_labels)

            spanned_labels = 0    # Clear

            offset = round(E_x_init)
            centered_labels = self.labels - offset
            centered_features = self.features - offset
            centered_modulo_labels = centered_labels % self.modulo_param
            centered_modulo_features = centered_features % self.modulo_param
            processed_labels = np.zeros(shape=centered_modulo_labels.shape)
            processed_features = np.zeros(shape=centered_modulo_features.shape)
            actual_map = {}
            for i in range(self.modulo_param):
                new_val = self.harmonic_dist_for_lin_opt(i)
                actual_map[i] = new_val
                mask_labels = (centered_modulo_labels == i) * 1
                mask_features = (centered_modulo_features == i) * 1
                mask_labels *= new_val
                mask_features *= new_val
                processed_labels += mask_labels
                processed_features += mask_features
            mul_for_cross_cov = processed_features * processed_labels
            processed_labels = np.repeat(processed_labels, self.repetitions)
            processed_features = np.repeat(processed_features, self.repetitions, axis=1)
            E_x = np.mean(processed_labels)
            E_y = np.mean(processed_features, axis=1)

            cov_y_y = np.zeros(shape=(processed_features.shape[0], processed_features.shape[0]))
            for i in range(processed_features.shape[0]):
                for j in range(processed_features.shape[0]):
                    curr_mean1 = np.mean(processed_features[i] * processed_features[j])
                    curr_mean2 = E_y[i] * E_y[j]
                    cov_y_y[i, j] = curr_mean1 - curr_mean2
            inv_cov_y_y = np.linalg.inv(cov_y_y)

            processed_labels = 0  # Clear
            processed_features = 0  # Clear

            mul_for_cross_cov_spanned = np.repeat(mul_for_cross_cov, self.repetitions, axis=1)
            cross_cov_x_y = np.mean(mul_for_cross_cov_spanned, axis=1)

            self.estimation_map = actual_map
            self.E_x_mapped = E_x
            self.E_y_mapped = E_y
            self.cross_cov_x_y_mapped = cross_cov_x_y
            self.inv_cov_y_y_mapped = inv_cov_y_y
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Statistics constructed in " + str(round(calc_time, 2)) + " seconds")

    def estimate(self, y):
        y_mod = [item % self.modulo_param for item in y]
        y_mapped = np.asarray([self.estimation_map[item] for item in y_mod], dtype=np.float64)
        centered_y_mapped = y_mapped - self.E_y_mapped
        mat_mul1 = np.matmul(self.inv_cov_y_y_mapped, centered_y_mapped)
        mat_mul2 = np.matmul(self.cross_cov_x_y_mapped, mat_mul1)
        estimator = self.E_x_mapped + mat_mul2
        return estimator

    def dump_statistics(self):
        if self.mode == 0:    # Estimating scalar x from vector y
            estimation_map_lst = [0 for i in range(self.modulo_param)]
            estimation_map_vec = np.asarray(estimation_map_lst)
            for i in range(estimation_map_vec.shape[0]):
                estimation_map_vec[i] = self.estimation_map[i]
            np.savetxt("map_as_vec.csv", estimation_map_vec, delimiter=",")
            np.savetxt("E_x_mapped.csv", np.asarray([self.E_x_mapped, 0]), delimiter=",")   # Ignore the 2nd entry
            np.savetxt("E_y_mapped.csv", self.E_y_mapped, delimiter=",")
            np.savetxt("cross_cov_x_y_mapped.csv", self.cross_cov_x_y_mapped, delimiter=",")
            np.savetxt("inv_cov_y_y_mapped.csv", self.inv_cov_y_y_mapped, delimiter=",")
            print("Finished dumping, rerun the code with sys.argv[5] = 0 to fix audio")

    def load_statistics(self, mode, modulo_param=12):
        self.mode = mode
        if self.mode == 0:    # Estimating scalar x from vector y
            self.modulo_param = modulo_param
            estimation_map_vec = np.loadtxt("map_as_vec.csv", delimiter=",")
            E_x_mapped = np.loadtxt("E_x_mapped.csv", delimiter=",")
            E_x_mapped = E_x_mapped[0]
            E_y_mapped = np.loadtxt("E_y_mapped.csv", delimiter=",")
            cross_cov_x_y_mapped = np.loadtxt("cross_cov_x_y_mapped.csv", delimiter=",")
            inv_cov_y_y_mapped = np.loadtxt("inv_cov_y_y_mapped.csv", delimiter=",")
            estimation_map = {}
            inverse_estimation_map = {}
            for idx, item in enumerate(estimation_map_vec):
                estimation_map[idx] = item
                inverse_estimation_map[item] = idx
            self.estimation_map = estimation_map
            self.inverse_estimation_map = inverse_estimation_map
            self.E_x_mapped = E_x_mapped
            self.E_y_mapped = E_y_mapped
            self.cross_cov_x_y_mapped = cross_cov_x_y_mapped
            self.inv_cov_y_y_mapped = inv_cov_y_y_mapped

    def fix_audio(self, up_down_feature_lst_lst, base_note=60, select_init_state=False):
        ret_lst = []
        last_note = base_note
        if select_init_state:
            state = select_init_state
        else:
            state = np.random.randint(0, 11, size=self.E_y_mapped.shape).tolist()
        for idx, item in enumerate(up_down_feature_lst_lst):
            if item == 666:
                ret_lst.append(666)
            elif item == 0:
                note_to_append = last_note
                ret_lst.append(note_to_append)
                temp_var = state.pop(0)
                state.append(state[-1])
            else:
                estimation = round(self.estimate(state))
                if item == 1:
                    if (last_note + self.inverse_estimation_map[estimation]) < self.midi_range:
                        note_to_append = last_note + self.inverse_estimation_map[estimation]
                        ret_lst.append(note_to_append)
                    else:
                        note_to_append = last_note - self.inverse_estimation_map[estimation]
                        ret_lst.append(note_to_append)
                elif item == -1:
                    if (last_note - self.inverse_estimation_map[estimation]) >= 0:
                        note_to_append = last_note - self.inverse_estimation_map[estimation]
                        ret_lst.append(note_to_append)
                    else:
                        note_to_append = last_note + self.inverse_estimation_map[estimation]
                        ret_lst.append(note_to_append)
                else:
                    print("This line should not execute - debug to find why it is")
                last_note = note_to_append
                temp_var = state.pop(0)
                state.append(estimation)
        return tuple(ret_lst)

