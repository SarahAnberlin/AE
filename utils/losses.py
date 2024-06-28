import torch.nn as nn
import torch
import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar):
    # 计算均方误差损失
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # 计算 KL 散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 返回总损失，可以调整两部分的权重
    return MSE + KLD
