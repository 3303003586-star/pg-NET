import numpy as np

def data_augmentation(x, y):
    if np.random.rand() < 0.25:
        x = np.fliplr(x)
        y = np.fliplr(y)

    # 随机确定是否垂直翻转
    if np.random.rand() < 0.25:
        x = np.flipud(x)
        y = np.flipud(y)

    # 随机确定是否对角线翻转
    if np.random.rand() < 0.25:
        x = np.rot90(x)
        y = np.rot90(y)

    if np.random.rand() < 0.25:
        x = x
        y = y

    return x, y