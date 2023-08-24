import numpy as np


# def bn_forward_naive(x, gamma, beta, running_mean, running_var, mode = "trian", eps = 1e-5, momentum = 0.9):
# 	n, ic, ih, iw = x.shape
# 	out = np.zeros(x.shape)
# 	if mode == 'train':
# 		batch_mean = np.zeros(running_mean.shape)
# 		batch_var = np.zeros(running_var.shape)
# 		for i in range(ic):
# 			batch_mean[i] = np.mean(x[:, i, :, :])
# 			batch_var[i] = np.sum((x[:, i, :, :] - batch_mean[i]) ** 2 ) / (n * ih * iw)
# 		for i in range(ic):
# 			out[:, i, :, :] = (x[:, i, :, :] - batch_mean[i]) / np.sqrt(batch_var[i] + eps)
# 			out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]
# 		#update
# 		running_mean = running_mean * momentum + batch_mean * (1 - momentum)
# 		running_var = running_var * momentum + batch_var * (1 - momentum)
# 	elif mode == 'test':
# 		for i in range(ic):
# 			out[:, i, :, :] = (x[:, i, :, :] - running_mean[i]) / np.sqrt(running_var[i] + eps)
# 			out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]
# 	else:
# 		raise ValueError('Invalid forward BN mode: %s' % mode)
# 	return out

# 参数：
#     - x: 输入张量，形状为 (N, C, H, W)
#     - gamma: 缩放参数，形状为 (C,)
#     - beta: 平移参数，形状为 (C,)
#     - running_mean: 运行均值，形状为 (C,)
#     - running_var: 运行方差，形状为 (C,)
#     - mode: 操作模式，可选值为 "train" 或 "test"
#     - momentum: 动量参数，用于更新运行均值和方差，默认为 0.9
#     - eps: 一个小的常数，用于避免除以零，默认为 1e-5
def BN(x, gamma, beta, running_mean, running_var, mode="train", momentum=0.9, eps=1e-5):
    n, c, h, w = x.shape
    out = np.zeros(shape=x.shape)
    if mode == "train":
        batch_mean = np.zeros(running_mean.shape)
        batch_var = np.zeros(running_var.shape)
        for i in range(c):
            print(x)
            # 计算批量期望与方差
            batch_mean[i] = np.mean(x[:, i, :, :])
            batch_var[i] = np.sum((x[:, i, :, :] - batch_mean) ** 2) / (n * h * w)
        for i in range(c):
            # 标准化
            out[:, i, :, :] = (x[:, i, :, :] - batch_mean[i]) / np.sqrt(batch_var[i] + eps)
            # 求BN值，让参数分布尽可能靠近标准正态分布
            out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]

        # 对运行的期望与方差进行更新
        running_mean = running_mean * momentum + batch_mean * (1 - momentum)
        running_var = running_var * momentum + batch_var * (1 - momentum)



    # 若是测试模式，则模型的参数减去对应的通道参数归一化，并且将其进行缩放与平移操作，使其最终的分布尽可能靠近正态分布
    elif mode == "test":
        for i in range(c):
            out[:, i, :, :] = (x[:, i, :, :] - running_mean[i]) / np.sqrt(running_var[i] + eps)
            out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]

    else:
        raise ValueError("Invalid foward BN mode: %s" % mode)

# if __name__ == '__main__':
# BN(x=100, gamma=0.5, running_mean=)
