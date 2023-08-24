import torch
import numpy as np
def bn_forward_naive(x, run_mean, run_var, gamma, beta, mode, momentum=0.9, eps= 0.0001):
    n, c, h,w = x.shape
    out = np.zeros(shape=x.shape)

    if mode == 'train':
        batch_mean = np.zeros(shape=run_mean.shape)
        batch_var = np.zeros(shape=run_var.shape)
        for i in range(c):
            # 计算期望、方差
            batch_mean = np.sum(x[:, i, :, :]) / n
            batch_var = np.sum((x[:, i, :, :] - batch_mean) ** 2) / (n * h * w)


        for i in range(c):
            # 归一化
            out[:, i, :, :] = np.sum((x[:, i, :, :] - batch_mean[:, i, :, :]) ** 2) / np.sqrt(eps + batch_var)
            # 缩放平移
            out[:, i, :, :] = gamma * out[:, i, :, :] + beta

        # 对期望与方差进行更新
        run_mean = run_mean * momentum + (1 - momentum ) * batch_mean
        run_var = run_var * momentum + (1 - momentum) * batch_var


    if mode == 'test':
        for i in range(c):
            # 归一化
            out[:, i, :, :] = np.sum((x[:, i, :, :] - run_mean[:, i, :, :]) ** 2) / np.sqrt(eps + run_var)
            # 缩放平移
            out[:, i, :, :] = gamma * out[:, i, :, :] + beta


    return out[c], run_mean, run_var


if __name__ == '__main__':
    x = np.array([3, 3, 3, 3])
    bn_forward_naive(x, 3, 3, 0.5, 0.5, mode='train')














