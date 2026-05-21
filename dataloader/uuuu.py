from __future__ import print_function, division
import re
import os
import math
import cv2
import torch
import random
import warnings
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from dataloader.split_str import extract_numbers_from_string
# from .data_augmentation import data_augmentation
from torchvision import transforms, utils, datasets, models

warnings.filterwarnings("ignore")

def compute_los_effect(reflectance_db, transmittance_db, distance):
    epsilon = 1e-6
    los_effect = np.exp(-(reflectance_db + transmittance_db) / (distance + epsilon))
    return los_effect

class IRM_Data(Dataset):
    def __init__(self, phase='train',
                 data=r"C:\Users\陈琦\Desktop\python\ICASSP/dataset/",
                 task='task2',
                 train_ratio=0.9,
                 seed=2025,
                 alpha=[2.15, 1.66, 1.66],
                 freq=[0.868e9, 1.8e9, 3.5e9],
                 c=3e8,
                 sample_rate=[0.0002],
                 size=256,
                 augment_mode=None,
                 random_augment=0,
                 transform=transforms.ToTensor()):

        self.inputs = []
        self.outputs = []
        self.c = c
        self.data = data
        self.task = task
        self.seed = seed
        self.alpha = alpha
        self.phase = phase
        self.freq = freq
        self.size = size
        self.transform = transform
        self.train_ratio = train_ratio
        self.sample_rate = sample_rate
        self.augment_mode = augment_mode
        self.random_augment = random_augment

        if self.task == 'task1':
            self.inputs_path = self.data + "Inputs/Task_1_ICASSP/"
            self.outputs_path = self.data + "Outputs/Task_1_ICASSP/"

        elif self.task == 'task2':
            self.inputs_path = self.data + "Inputs/Task_2_ICASSP/"
            self.outputs_path = self.data + "Outputs/Task_2_ICASSP/"

        elif self.task == 'task3':
            self.inputs_path = self.data + "Inputs/Task_3_ICASSP/"
            self.outputs_path = self.data + "Outputs/Task_3_ICASSP/"

        self.TX_path = self.data + "Positions/"
        self.build_path = self.data + "Building_Details/"
        self.pattern = self.data + "Radiation_Patterns/"

        # read the data path
        for ext in ['*.png']:
            self.inputs.extend(glob(os.path.join(self.inputs_path, ext)))

        for ext in ['*.png']:
            self.outputs.extend(glob(os.path.join(self.outputs_path, ext)))

        # shuffle the data path
        random.seed(self.seed)
        combined = list(zip(self.inputs, self.outputs))
        random.shuffle(combined)

        list1_shuffled, list2_shuffled = zip(*combined)

        self.inputs = list(list1_shuffled)
        self.outputs = list(list2_shuffled)

        # split the data path

        if self.phase == 'train':
            self.inputs = self.inputs[:int(len(self.inputs) * self.train_ratio)]
            self.outputs = self.outputs[:int(len(self.outputs) * self.train_ratio)]

        elif self.phase == 'val':
            self.inputs = self.inputs[int(len(self.inputs) * self.train_ratio):]
            self.outputs = self.outputs[int(len(self.outputs) * self.train_ratio):]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        # Loading input features
        inputs = Image.open(self.inputs[index]).convert('RGB')
        arr_inputs = np.asarray(inputs)
        size = arr_inputs.shape

        idx = extract_numbers_from_string(self.inputs[index])

        # Loading building features (.csv)
        building_details = pd.read_csv(self.build_path + "B" + str(idx[1]) + "_Details.csv")
        W, H = building_details["W"].iloc[0], building_details["H"].iloc[0]

        # Loading antenna features (.csv)
        sampling_positions = pd.read_csv(self.TX_path + "Positions_B" + str(idx[1]) + "_Ant" + str(idx[2]) + "_f" + str(idx[3]) + '.csv')

        x_ant = sampling_positions["Y"].loc[idx[4]]
        y_ant = sampling_positions["X"].loc[idx[4]]

        X_points = np.repeat(np.linspace(0, W - 1, W), H, axis=0).reshape(W, H).transpose()
        Y_points = np.repeat(np.linspace(0, H - 1, H), W, axis=0).reshape(H, W)

        angles = -(180 / np.pi) * np.arctan2((y_ant - Y_points), (x_ant - X_points)) + 180 + \
        sampling_positions['Azimuth'].iloc[idx[4]]

        angles = np.where(angles > 359, angles - 360, angles).astype(int)

        # loading antenna pattern
        antenna_azimuth_pattern = np.genfromtxt(self.pattern + "Ant" + str(idx[2]) + "_Pattern" + ".csv", delimiter=',', skip_header=1)

        # antenna gain
        angles = np.where(angles > 358, 358, angles).astype(int)
        g = antenna_azimuth_pattern[angles]

        # loading freq
        if idx[3] == 1:
            fre = np.ones((size[0], size[1])) * self.freq[0]
        elif idx[3] == 2:
            fre = np.ones((size[0], size[1])) * self.freq[1]
        else:
            fre = np.ones((size[0], size[1])) * self.freq[2]


        # Image name
        name = "B" + str(idx[1]) + "_Ant" + str(idx[2]) + "_f" + str(idx[3]) + "_S" + str(idx[4]) + '.png'

        # Loading target
        outputs = Image.open(self.outputs[index]).convert('L')
        arr_outputs = np.asarray(outputs)
        out_s = arr_outputs

        # loading sample point

        arr_sample = np.zeros((H, W))
        random_rate = np.random.choice(self.sample_rate)
        sample_points = np.random.choice([0, 1], size=(H, W), p=[1 - random_rate, random_rate])
        arr_sample[sample_points == 1] = 1
        arr_sample = arr_sample * out_s

        # 计算LOS效应
        reflectance_db = arr_inputs[:, :, 0].astype(np.float32)
        transmittance_db = arr_inputs[:, :, 1].astype(np.float32)
        distance = arr_inputs[:, :, 2].astype(np.float32)
        los_effect = compute_los_effect(reflectance_db, transmittance_db, distance)

        min_index_flat = np.argmin(arr_inputs[:, :, 2])
        shape = arr_outputs.shape
        min_index = np.unravel_index(min_index_flat, shape)
        image_TX = np.zeros((size[0], size[1]))
        image_TX[min_index[0], min_index[1]] = 1

        distance_to_transmitter = np.sqrt((x_ant - X_points) ** 2 + (y_ant - Y_points) ** 2) * 0.25

        freq = self.freq[idx[3] - 1]
        PL_matrix = 20 * np.log10(distance_to_transmitter) + 20 * np.log10(freq) + 20 * np.log10(4 * np.pi / self.c)
        PL_matrix[np.isinf(PL_matrix)] = 0

        los_effect = np.expand_dims(los_effect, axis=-1)
        image_TX = np.expand_dims(image_TX, axis=-1)
        PL_matrix = np.expand_dims(PL_matrix, axis=-1)
        angles = np.expand_dims(angles, axis=-1)
        sample = np.expand_dims(arr_sample, axis=-1)
        g, fre = np.expand_dims(g, axis=-1), np.expand_dims(fre, axis=-1)

        in_s = np.concatenate((arr_inputs, sample, los_effect,image_TX, PL_matrix, angles, g), axis=-1)

        # Data augmentation
        if self.phase == 'train':
            # in_s, out_s = data_augmentation(in_s, out_s)
            in_s, out_s = self._apply_random_augment(in_s, out_s)
            # print(in_s.shape) (314, 357)

            # To tensor
        in_s = cv2.resize(in_s, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)
        arr_in = self.transform((in_s / 255).copy()).type(torch.float32)
        out_s = cv2.resize(out_s, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)
        arr_label = self.transform((out_s / 255).copy()).type(torch.float32)

        return arr_in, arr_label, [W, H], name

    def _apply_augment(self, X, y):
        if self.augment_mode == 'original':
            return X, y
        elif self.augment_mode == 'rotate_90':
            X = np.rot90(X, k=1, axes=(0, 1))
            y = np.rot90(y, k=1, axes=(0, 1))
        elif self.augment_mode == 'rotate_180':
            X = np.rot90(X, k=2, axes=(0, 1))
            y = np.rot90(y, k=2, axes=(0, 1))
        elif self.augment_mode == 'rotate_270':
            X = np.rot90(X, k=3, axes=(0, 1))
            y = np.rot90(y, k=3, axes=(0, 1))
        elif self.augment_mode == 'flip_y':
            X = np.flip(X, axis=1)
            y = np.flip(y, axis=1)
        elif self.augment_mode == 'flip_x':
            X = np.flip(X, axis=0)
            y = np.flip(y, axis=0)
        elif self.augment_mode == 'flip_x_rotate_90':
            X = np.flip(X, axis=0)
            X = np.rot90(X, k=1, axes=(0, 1))
            y = np.flip(y, axis=0)
            y = np.rot90(y, k=1, axes=(0, 1))
        elif self.augment_mode == 'flip_y_rotate_90':
            X = np.flip(X, axis=1)
            X = np.rot90(X, k=1, axes=(0, 1))
            y = np.flip(y, axis=1)
            y = np.rot90(y, k=1, axes=(0, 1))

        return X, y

    def _apply_random_augment(self, X, y):
        augmentation_modes = ['original', 'rotate_90', 'rotate_180', 'rotate_270',
                              'flip_y', 'flip_x', 'flip_x_rotate_90', 'flip_y_rotate_90']
        chosen_mode = random.choice(augmentation_modes)

        original_mode = self.augment_mode
        self.augment_mode = chosen_mode

        X_aug, y_aug = self._apply_augment(X, y)

        self.augment_mode = original_mode
        return X_aug, y_aug

def test():
    dataset = IRM_Data(phase='train')
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    for x, y, z, w in loader:
        print(x.shape, y.shape, w)

if __name__ == "__main__":
    test()