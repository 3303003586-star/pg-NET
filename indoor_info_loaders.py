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
from skimage import io, transform, morphology
from torch.utils.data import Dataset, DataLoader
from .split_str import extract_numbers_from_string
from torchvision import transforms, utils, datasets, models

warnings.filterwarnings("ignore")

class IRM_Data(Dataset):
    def __init__(self, phase='train',
                 data="dataset/",
                 task='task1',
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
        self.norm_min = -160.0
        self.norm_max = -13.0
        self.n_los = 2.013
        self.lw_fixed = 3.8
        self.pixel_per_meter = 4
        self.n_nlos_map = {1: 4.1562, 2: 5.1641, 3: 6.5155}  # 频率差异化 n 值

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
        # 1. 加载输入特征图像 (RGB)
        inputs = Image.open(self.inputs[index]).convert('RGB')
        arr_inputs = np.asarray(inputs)
        h_orig, w_orig, _ = arr_inputs.shape  # 获取原始图高宽

        # 2. 提取 ID 信息
        idx = extract_numbers_from_string(self.inputs[index])
        # idx[1]: Building ID, idx[4]: Station ID (S0, S1...)

        # 3. 加载建筑细节和基站坐标 (保持原逻辑)
        building_details = pd.read_csv(self.build_path + "B" + str(idx[1]) + "_Details.csv")
        W, H = building_details["W"].iloc[0], building_details["H"].iloc[0]

        sampling_positions = pd.read_csv(
            self.TX_path + "Positions_B" + str(idx[1]) + "_Ant1"  + "_f"  + str(idx[3]) + '.csv')
        x_ant = sampling_positions["Y"].loc[idx[4]]
        y_ant = sampling_positions["X"].loc[idx[4]]

        # 4. 加载计算好的 3750 张无线电地图 (取代旧的自由空间模型)
        # 假设你的 npy 存在 self.data + "PL_Maps_MWM/" 目录下
        # 文件命名格式参考: PL_B1_BS0.npy
        #
        pl_maps_dir = os.path.join("E:\edge load\ICASSP2025_Dataset\ICASSP2025_Dataset\line\smooth00")
        #pl_maps_dir = os.path.join(self.data + "smooth00")
        name_npy = f"PL_Map_B{idx[1]}_S{idx[4]}.npy"

        npy_path = os.path.join(pl_maps_dir, name_npy)

        if os.path.exists(npy_path):
            # 直接加载 pre-computed 的多墙模型损耗图
            PL_matrix = np.load(npy_path)
        # # # else:
        # # #     # 如果文件缺失，回退到 0（或抛出错误提醒）
        # # #     print(f"Warning: {npy_path} not found!")
        # PL_matrix = np.zeros((H, W))



        # # --- 4. 核心集成 ---
        # # A. 提取墙体骨架 (基于当前 RGB 输入)
        # gray = cv2.cvtColor(arr_inputs, cv2.COLOR_RGB2GRAY)
        # grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3)))
        # _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # skel = morphology.skeletonize(binary > 127)
        # skel_img = (skel.astype(np.uint8) * 255)  # 255 代表墙体
        #
        # # B. 距离场计算
        # x_coords, y_coords = np.meshgrid(np.arange(w_orig), np.arange(h_orig))
        # dist_px = np.sqrt((x_coords - x_ant) ** 2 + (y_coords - y_ant) ** 2)
        # dist_m = np.maximum(dist_px / self.pixel_per_meter, 0.1)
        #
        # # 判定 LOS
        # los_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        #
        # for step_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        #     check_x = np.clip(x_ant + (x_coords - x_ant) * step_ratio, 0, w_orig - 1).astype(int)
        #     check_y = np.clip(y_ant + (y_coords - y_ant) * step_ratio, 0, h_orig - 1).astype(int)
        #     # 如果路径上的采样点碰到了墙，则该位置不是 LOS
        #     # 注意：这里逻辑需要反向思维，默认设为 LOS，碰墙剔除
        #
        # # 推荐使用更高效的批量射线判定逻辑：
        # los_mask = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
        # # 采样 10 个点来检测路径是否有墙（在线运行的折中方案）
        # for t in np.linspace(0.1, 0.9, 10):
        #     lx = np.clip(x_ant + (x_coords - x_ant) * t, 0, w_orig - 1).astype(int)
        #     ly = np.clip(y_ant + (y_coords - y_ant) * t, 0, h_orig - 1).astype(int)
        #     los_mask[skel_img[ly, lx] > 127] = 0
        #
        # # D. 5px 去毛刺
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # los_mask_clean = cv2.morphologyEx(los_mask, cv2.MORPH_OPEN, kernel)
        #
        # # E. 连续型路径损耗计算 (按照你最新的 += 叠加逻辑)
        # f_idx = idx[3]
        # freq_val = self.freq[f_idx - 1]
        # current_n_nlos = self.n_nlos_map[f_idx]
        #
        # # 1. 计算基础 LOS 损耗
        # pl_map = 10 * self.n_los * np.log10(dist_m) + 20 * math.log10(freq_val) - 147.55
        #
        # # 2. 叠加 NLOS 修正
        # nlos_idx = (los_mask_clean == 0)
        # n_diff = current_n_nlos - self.n_los
        # pl_map[nlos_idx] += self.lw_fixed + 10 * n_diff * np.log10(dist_m[nlos_idx])
        #
        # PL_matrix = pl_map.astype(np.float32)
        # # # # PL_matrix = np.zeros_like(pl_map, dtype=np.float32) # 将整个通道置为空（全0）
        # # # # PL_matrix[:]=0






        # 5. 处理角度和天线增益 (保持原逻辑)
        X_points = np.repeat(np.linspace(0, W - 1, W), H, axis=0).reshape(W, H).transpose()
        Y_points = np.repeat(np.linspace(0, H - 1, H), W, axis=0).reshape(H, W)
        angles = -(180 / np.pi) * np.arctan2((y_ant - Y_points), (x_ant - X_points)) + 180 + \
                 sampling_positions['Azimuth'].iloc[idx[4]]
        angles = np.where(angles > 359, angles - 360, angles).astype(int)

        antenna_azimuth_pattern = np.genfromtxt(self.pattern + "Ant" + str(idx[2]) + "_Pattern" + ".csv", delimiter=',',
                                                skip_header=1)
        angles_clipped = np.where(angles > 358, 358, angles).astype(int)

        # g = antenna_azimuth_pattern[angles_clipped]

        # 6. 加载频率特征 (保持原逻辑)
        # if idx[3] == 1:
        #     fre = np.ones((H, W)) * self.freq[0]
        # elif idx[3] == 2:
        #     fre = np.ones((H, W)) * self.freq[1]
        # else:
        #     fre = np.ones((H, W)) * self.freq[2]

        current_freq_val = self.freq[idx[3] - 1]
        fre_matrix = np.ones((H, W), dtype=np.float32) * current_freq_val
        fre_matrix_norm = (fre_matrix / 3.5e9)*255  # 以最大频率 3.5GHz 归一化

        # 7. 加载目标 Label
        outputs = Image.open(self.outputs[index]).convert('L')
        out_s = np.asarray(outputs)

        # 8. 基站位置处理：从 One-hot 点变为高斯热力点
        image_TX = np.zeros((H, W))
        min_index_flat = np.argmin(arr_inputs[:, :, 2])
        min_index = np.unravel_index(min_index_flat, (H, W))
        image_TX[min_index[0], min_index[1]] =1  # 先设为最大值

        # 使用高斯模糊平滑基站位置，减少“白点”效应
        # ksize 必须为奇数，sigma 越大圆点越弥散


        # 9. 采样点加载 (保持原逻辑)
        # 随机

        arr_sample = np.zeros((H, W))
        random_rate = np.random.choice(self.sample_rate)
        sample_mask = np.random.choice([0, 1], size=(H, W), p=[1 - random_rate, random_rate])
        arr_sample[sample_mask == 1] = out_s[sample_mask == 1]


        # # 远端
        # random_rate = np.random.choice(self.sample_rate)
        # num_samples = int(H * W * random_rate)
        #
        # # all position coordinate
        # yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        #
        # # TX position
        # cy, cx = y_ant, x_ant
        #
        # # distance
        # dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        #
        # # 采样位置大于最小边长的1/2倍
        # min_dist = min(H, W) / 2
        # valid_mask = dist > min_dist
        # valid_indices = np.argwhere(valid_mask)
        #
        # # 查看数量是否足够
        # if len(valid_indices) < num_samples:
        #     raise ValueError("The number of sampling points is insufficient")
        #
        # # 选择所需要的点
        # selected_indices = valid_indices[np.random.choice(len(valid_indices), num_samples, replace=False)]
        #
        # # 赋值到全零矩阵中
        # arr_sample = np.zeros((H, W), dtype=out_s.dtype)
        # for y, x in selected_indices:
        #     arr_sample[y, x] = out_s[y, x]



        # 2-8采样
        # # 1. 基础参数与坐标准备
        # random_rate = np.random.choice(self.sample_rate)
        # num_samples = int(H * W * random_rate)
        #
        # yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        # cy, cx = y_ant, x_ant
        # dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        #
        # # 定义分区边界
        # min_dist = min(H, W) / 2
        # near_mask = dist <= min_dist
        # far_mask = dist > min_dist
        #
        # near_indices = np.argwhere(near_mask)
        # far_indices = np.argwhere(far_mask)
        #
        # # 2. 动态调整分配比例，防止点位过密
        # # 设定一个安全上限：采样点不能超过该区域总像素的 5%，否则就会出现“小格感”
        # density_limit = 0.05
        #
        # num_near = int(num_samples * 0.2)
        # num_far = num_samples - num_near
        #
        # # 检查远端点是否过密，如果过密，则强制削减并重新分配
        # if num_far > len(far_indices) * density_limit:
        #     num_far = int(len(far_indices) * density_limit)
        #     num_near = num_samples - num_far
        #
        # # 3. 核心改进：打乱顺序后进行分区抽样
        # # 使用 np.random.permutation 确保空间上的随机性，避免聚集
        # if len(near_indices) >= num_near and len(far_indices) >= num_far:
        #     idx_near = np.random.choice(len(near_indices), num_near, replace=False)
        #     idx_far = np.random.choice(len(far_indices), num_far, replace=False)
        #
        #     selected_near = near_indices[idx_near]
        #     selected_far = far_indices[idx_far]
        #     selected_indices = np.vstack((selected_near, selected_far))
        # else:
        #     # 兜底方案：全图随机
        #     all_indices = np.argwhere(np.ones((H, W)))
        #     selected_indices = all_indices[np.random.choice(len(all_indices), num_samples, replace=False)]
        #
        # # 4. 赋值（保持 0 填充，只有采样点有值）
        # arr_sample = np.zeros((H, W), dtype=out_s.dtype)
        # for y, x in selected_indices:
        #     arr_sample[y, x] = out_s[y, x]



        # 10. 维度扩展并拼接 (保持 8 通道)
        image_TX = np.expand_dims(image_TX, axis=-1)
        PL_matrix = np.expand_dims(PL_matrix, axis=-1)
        angles = np.expand_dims(angles.astype(np.float32), axis=-1)
        sample = np.expand_dims(arr_sample.astype(np.float32), axis=-1)
        # g = np.expand_dims(g.astype(np.float32), axis=-1)
        fre_matrix_norm = np.expand_dims(fre_matrix_norm.astype(np.float32), axis=-1)

        # 拼接顺序：RGB(3) + Sample(1) + TX(1) + PL_MWM(1) + Angles(1) + Gain(1) = 8通道

        in_s = np.concatenate((
            # 现在它是索引 0 (0:1)
            arr_inputs,
            sample,
            image_TX,
            PL_matrix,
            angles,
            fre_matrix_norm
        ), axis=-1)
        # 11. 数据增强与 Resize (归一化)
        if self.phase == 'train':
            in_s, out_s = self._apply_random_augment(in_s, out_s)

        in_s = cv2.resize(in_s, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)
        out_s = cv2.resize(out_s, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)

        # 最终归一化转 Tensor
        arr_in = self.transform((in_s / 255.0).copy()).type(torch.float32)
        arr_label = self.transform((out_s / 255.0).copy()).type(torch.float32)
        name = "B" + str(idx[1]) + "_Ant1"  + "_f"+ str(idx[3]) + "_S" + str(idx[4]) + '.png'
        return arr_in, arr_label, [W, H], name

    def _apply_augment(self, X, y):
        augmentation_options = {
            'original': lambda x: x,
            'rotate_90': lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            'rotate_180': lambda x: cv2.rotate(x, cv2.ROTATE_180),
            'rotate_270': lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
            'flip_y': lambda x: cv2.flip(x, 1),
            'flip_x': lambda x: cv2.flip(x, 0),
            'flip_x_rotate_90': lambda x: cv2.rotate(cv2.flip(x, 0), cv2.ROTATE_90_CLOCKWISE),
            'flip_y_rotate_90': lambda x: cv2.rotate(cv2.flip(x, 1), cv2.ROTATE_90_CLOCKWISE)
        }

        if self.augment_mode in augmentation_options:
            X_1 = augmentation_options[self.augment_mode](X[:, :, :4])
            X_2 = augmentation_options[self.augment_mode](X[:, :, 4:])

            # print(X_1.shape, X_2.shape)
            # X = augmentation_options[self.augment_mode](X)

            X = np.dstack((X_1, X_2))
            y = augmentation_options[self.augment_mode](y)

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
    dataset = IRM_Data(phase='val')
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    for x, y, z, w in loader:
        print(x.shape, y.shape, w)

if __name__ == "__main__":
    test()