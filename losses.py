import torch.nn.functional as F

def custom_loss(output, target):
    return F.l1_loss(output, target, reduction='mean')  # 计算整个矩阵的 MAE