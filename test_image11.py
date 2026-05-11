import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from model import uuuu
from dataloader import indoor_info_loaders
from torchmetrics import R2Score
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# loading model
device = torch.device("cuda:0")
model = uuuu.MyModel()
model.load_state_dict(torch.load('model_result/itu1002-2.pt'))
model.to(device)

def main_worker():

    # loading test data
    test_data = indoor_info_loaders.IRM_Data(phase='val')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    err1 = []
    err2 = []
    start_time = time.time()
    for x, target, size, img_name in test_dataloader:

        h, w = size[0].tolist(),  size[1].tolist()  # 先转列表再解包
        x, target = (x.to(device), target.to(device))

        interation += 1

        with torch.no_grad():
            pre = model(x)

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        image = im * 255
        image = cv2.resize(image, (h[0], w[0]), interpolation=cv2.INTER_LANCZOS4)
        images = Image.fromarray(image.astype(np.uint8))

        # predict
        test = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        test = test.squeeze(0)
        test = test.squeeze(0)
        im1 = test.cpu().numpy()
        predict = im1 * 255
        predict = cv2.resize(predict, (h[0], w[0]), interpolation=cv2.INTER_LANCZOS4)
        predict1 = Image.fromarray(predict.astype(np.uint8))

        # calculate rmse
        rmse1 = np.sqrt(np.mean((im - im1) ** 2))
        err1.append(rmse1)
        # calculate nmse
        nmse1 = np.mean((im - im1) ** 2)/np.mean((0 - im) ** 2)
        err2.append(nmse1)

        # 保存
        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join("image_result", f'{image_name}_target.png'))
        predict1.save(os.path.join("image_result", f'{image_name}_predict1.png'))
        print(f'saving to {os.path.join("image_result", image_name)}', "RMSE:", rmse1, "NMSE:", nmse1)

        # the number total of 8000
        if interation >= 8000:
            break

    end_time = time.time()
    runtime = end_time - start_time
    print("Total runtime: {:.2f} seconds".format(runtime))

    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)

    print('一阶段测试集均方根误差：', rmse_err)
    print('一阶段测试集归一化均方误差：', nmse_err)

if __name__ == '__main__':
 main_worker()