from __future__ import print_function, division
import os
import time
import copy
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm
from model import uuuu
import torch.optim as optim
from dataloader import indoor_info_loaders
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def save_prediction_comparison(pred, target, epoch, save_dir="val_results"):
    """
        inputs: 原始 8 通道输入 (B, 8, 256, 256)
        pred: 模型输出 (B, 1, 256, 256)
        target: 真实标签 (B, 1, 256, 256)
        """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取第 0 通道：物理底图 MWM

    p = pred[0, 0].cpu().detach().numpy()
    t = target[0, 0].cpu().detach().numpy()

    # 计算残差 (模型到底在 MWM 上改了什么)


    fig, axes = plt.subplots(1, 2, figsize=(24, 6))




    # 模型最终预测
    axes[1].set_title("Final Prediction (MWM + Delta)")
    axes[1].imshow(p, cmap='viridis')

    # 真实标签
    axes[2].set_title("Ground Truth")
    axes[2].imshow(t, cmap='viridis')




    for ax in axes: ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_residual_analysis.png"))
    plt.close()


def get_gradient(x):
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    # 使用 F.pad 将尺寸补回 (B, C, 256, 256)
    # pad 参数顺序是 (左, 右, 上, 下)
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
    return grad_x, grad_y



def gradient_loss(pred, target):
    grad_x_pred, grad_y_pred = get_gradient(pred)
    grad_x_gt, grad_y_gt = get_gradient(target)
    loss_g = torch.mean(torch.abs(grad_x_pred - grad_x_gt)) + \
             torch.mean(torch.abs(grad_y_pred - grad_y_gt))
    return loss_g


def combine_loss(pred, target, metrics, alpha=0.8):
    # MSE Loss
    criterion = nn.MSELoss()
    mse_loss = criterion(pred, target)
    metrics['MSE_loss'] += mse_loss.item() * target.size(0)

    # SSIM Loss
    ssim_value = ssim(pred, target, data_range=1.0)
    ssim_loss = 1 - ssim_value
    metrics['SSIM_loss'] += ssim_loss.item() * target.size(0)

    loss = alpha * mse_loss + (1 - alpha) * ssim_loss
    metrics['combined_loss'] += loss.item() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=130):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0
            last_pred, last_target = None,None

            for x, target, size, name in tqdm(main_dataloaders[phase]):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(x)
                    loss = combine_loss(outputs, target, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        last_inputs,last_pred, last_target = x,outputs, target
                epoch_samples += target.size(0)

            print_metrics(metrics, epoch_samples, phase)
            current_val_mse = metrics['MSE_loss'] / epoch_samples
            current_combined_loss = metrics['combined_loss'] / epoch_samples

            # 核心修改：如果是验证阶段，更新 ReduceLROnPlateau
            if phase == 'val':
                # 监控 MSE_loss，因为它直接决定了你的 RMSE 目标
                scheduler.step(current_val_mse)
                for param_group in optimizer.param_groups:
                    print("Current Learning Rate: {:.6f}".format(param_group['lr']))

                # 保存对比图
                # if last_inputs is not None:
                #     save_prediction_comparison(last_pred, last_target, epoch)

                # 以验证集 MSE 作为保存最佳模型的依据
                if current_val_mse < best_loss:
                    print(f"Validation MSE decreased ({best_loss:.6f} --> {current_val_mse:.6f}). Saving model...")
                    best_loss = current_val_mse
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    from collections import defaultdict

    device = torch.device("cuda:0")
    warnings.filterwarnings("ignore")

    Radio_train = indoor_info_loaders.IRM_Data(phase="train", task="task1")
    Radio_val = indoor_info_loaders.IRM_Data(phase="val", task="task1")

    batch_size = 4
    main_dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }

    model = uuuu.MyModel().to(device)
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

    # 修改为基于验证集表现自动调整学习率的调度器
    # factor: 缩放因子，patience: 容忍多少个epoch指标不下降
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=5, verbose=True)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=120)

    if not os.path.exists('model_result'): os.mkdir('model_result')
    torch.save(model.state_dict(), 'model_result/itu1002-3.pt')