import torch
import numpy as np
import pickle
from collections import defaultdict

batch_size = 32

class DataGenerator:
    def __init__(self):
        print("Loading data...")
        self.map_size = (51,158)

        self.train_trajectories, self.train_sd, self.train_traj_num = self.build_dataset('train')
        self.test_trajectories, self.test_sd, self.test_traj_num = self.build_dataset('test')

    def build_dataset(self, data_type):
        data_name = "./data/processed_porto.csv".split('.')
        data_name[-2] += "_{}".format(data_type)
        data_name = ".".join(data_name)
        trajectories = sorted([
            eval(eachline) for eachline in open(data_name, 'r').readlines()
        ], key=lambda k: len(k))
        traj_num = len(trajectories)
        print("{} {} trajectories loading complete.".format(traj_num, data_type))

        traj_sd = defaultdict(list)
        for idx, traj in enumerate(trajectories):
            traj_sd[(traj[0], traj[-1])].append(idx)
        return trajectories, traj_sd, traj_num

    def batch_pad(self, batch_x):
        max_len = max(len(x) for x in batch_x)
        batch_encode_input = [x + [0] * (max_len - len(x)) for x in batch_x]
        batch_decode_input = [[8059] + x + [0] * (max_len - len(x)) for x in batch_x]
        batch_decode_output = [x + [8060] + [0] * (max_len - len(x)) for x in batch_x]
        return batch_encode_input, batch_decode_input, batch_decode_output

    def inject_outliers(self, data_type, ratio=0.05, level=5, point_prob=0.3, vary=False):
        if data_type == "train":
            out_filename = 'train_porto_outliers.pkl'
            traj_num = self.train_traj_num
            trajectories = self.train_trajectories
        elif data_type == "test":
            out_filename = 'test_porto_outliers.pkl'
            traj_num = self.test_traj_num
            trajectories = self.test_trajectories
        else:
            raise ValueError("data_type is not 'train' or 'test'.")
        size = int(traj_num * ratio)

        if data_type == "train":
            self.outlier_idx = selected_idx = np.random.choice(traj_num, size=size*2, replace=False)
            self.detour_idx = detour_idx = selected_idx[:size]
            self.switching_idx = switching_idx = selected_idx[size:]

        elif data_type == "test":
            self.outlier_idx = selected_idx = np.random.choice(traj_num, size=size*3, replace=False)
            self.detour_idx = detour_idx = selected_idx[:size]
            self.switching_idx = switching_idx = selected_idx[size:size*2]
            self.gps_idx = gps_idx = selected_idx[size*2:]

        detour_outliers = self.detour_batch([trajectories[idx] for idx in detour_idx],
                                    level=level, prob=point_prob)

        switching_outliers = self.switching_batch([trajectories[idx] for idx in switching_idx],
                                    level=level, prob=point_prob, vary=vary)
        if data_type == "test":
            gps_outliers = self.gps_batch([trajectories[idx] for idx in gps_idx],
                                    prob=point_prob)

        with open('./data/data_osr1/detour_' + out_filename, 'wb') as fp:
            pickle.dump(dict(zip(detour_idx, detour_outliers)), fp)

        with open('./data/data_osr1/switching_' + out_filename, 'wb') as fp:
            pickle.dump(dict(zip(switching_idx, switching_outliers)), fp)

        if data_type == "test":
            with open('./data/data_osr1/gps_' + out_filename, 'wb') as fp:
                pickle.dump(dict(zip(gps_idx, gps_outliers)), fp)

        print("{} detour outliers injection into {} is completed.".format(len(detour_outliers), data_type))
        print("{} switching outliers injection into {} is completed.".format(len(switching_outliers), data_type))
        if data_type == "test":
            print("{} gps anomaly outliers injection into {} is completed.".format(len(gps_outliers), data_type))

    def load_outliers(self, filename, data_type):

        if data_type == "train":
            trajectories = self.train_trajectories
            labels = [0 for i in range(self.train_traj_num)]
        elif data_type == "test":
            trajectories = self.test_trajectories
            labels = [0 for i in range(self.test_traj_num)]

        with open('./data/data_osr1/detour_' + filename, 'rb') as fp:
            detour_idx = pickle.load(fp)
        for idx, o in detour_idx.items():
            trajectories[idx] = o
            labels[idx] = 1

        with open('./data/data_osr1/switching_' + filename, 'rb') as fp:
            switching_idx = pickle.load(fp)
        for idx, o in switching_idx.items():
            trajectories[idx] = o
            labels[idx] = 2

        if data_type == "test":
            with open('./data/data_osr1/gps_' + filename, 'rb') as fp:
                    gps_idx = pickle.load(fp)
            for idx, o in gps_idx.items():
                trajectories[idx] = o
                labels[idx] = 3

        if data_type == "train":
            self.train_trajectories = trajectories
        elif data_type == "test":
            self.test_trajectories = trajectories

        self.labels = labels

        print("{} {} detour trajectories loading complete.".format(len(detour_idx), data_type))
        print("{} {} switching trajectories loading complete.".format(len(switching_idx), data_type))
        if data_type == "test":
            print("{} {} gps anomaly trajectories loading complete.".format(len(gps_idx), data_type))

    def iterate_data(self, data_type='test'):
        if data_type == 'train':
            traj_num = self.train_traj_num
            pad = batch_size - traj_num % batch_size
            trajectories = self.train_trajectories + self.train_trajectories[:pad]
            labels = self.labels + self.labels[:pad]
        elif data_type == 'test':
            traj_num = self.test_traj_num
            pad = batch_size - traj_num % batch_size
            trajectories = self.test_trajectories + self.test_trajectories[:pad]
            labels = self.labels + self.labels[:pad]
        for shortest_idx in range(0, traj_num, batch_size):
            longest_idx = shortest_idx + batch_size
            label = labels[shortest_idx:longest_idx]
            batch_trajectories = []
            for tid in range(shortest_idx, longest_idx):
                batch_trajectories.append(trajectories[tid])
            batch_seq_length = [len(traj) for traj in batch_trajectories]
            batch_encode_input, batch_decode_input, batch_decode_output = self.batch_pad(batch_trajectories)
            yield torch.LongTensor(batch_encode_input), torch.LongTensor(batch_decode_input), torch.LongTensor(batch_decode_output), torch.LongTensor(batch_seq_length), torch.LongTensor(label)

    #Detour
    def detour_batch(self, batch_x, level, prob):
        noisy_batch_x = []
        for traj in batch_x:
            noisy_batch_x.append([traj[0]] + [self._detour_point(p, level)
                                 if not p == 0 and np.random.random() < prob else p
                                 for p in traj[1:-1]] + [traj[-1]])
        return noisy_batch_x

    def _detour_point(self, point, level, offset=None):
        map_size = self.map_size
        x, y = int(point // map_size[1]), int(point % map_size[1])
        if offset is None:
            offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
            x += x_offset * level
            y += y_offset * level
        return int(x * map_size[1] + y)

    #Route-switching
    def switching_batch(self, batch_x, level, prob, vary=False):
        map_size = self.map_size
        noisy_batch_x = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for traj in batch_x:
            anomaly_len = int((len(traj) - 2) * prob)
            anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            offset = [int(traj[anomaly_st_loc] // map_size[1]) - int(traj[anomaly_ed_loc] // map_size[1]),
                      int(traj[anomaly_st_loc] % map_size[1]) - int(traj[anomaly_ed_loc] % map_size[1])]
            if offset[0] == 0: div0 = 1
            else: div0 = abs(offset[0])
            if offset[1] == 0: div1 = 1
            else: div1 = abs(offset[1])

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            noisy_batch_x.append(traj[:anomaly_st_loc] +
                                 [self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])
        return noisy_batch_x

    #Navigation
    def gps_batch(self, batch_x, prob):
        noisy_batch_x = []
        for traj in batch_x:
            anomaly_len = int(len(traj) * prob)
            anomaly_st_loc = np.random.randint(1, len(traj) - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            noisy_batch_x.append(traj[:anomaly_st_loc] +
                                 [np.random.randint(1, 8059) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])
        return noisy_batch_x