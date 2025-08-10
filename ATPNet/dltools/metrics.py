import numpy as np
import torch


def mse(preds, labels):
    loss = (preds - labels) ** 2
    return torch.mean(loss)


def rmse(preds, labels):
    return torch.sqrt(mse(preds, labels))


def mae(preds, labels):
    loss = torch.abs(preds - labels)
    return torch.mean(loss)


def mape(preds, labels):
    return masked_mape(preds, labels, 0.0)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)  # 不计算缺失值的损失
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def rrse(preds, labels):
    # preds = preds.view(-1, 1)
    # labels = labels.view(-1, 1)

    error = preds - labels
    scale = labels - torch.mean(labels)
    ee = torch.mm(error.transpose(1, 0), error).view(-1)
    ss = torch.mm(scale.transpose(1, 0), scale).view(-1)

    return torch.sqrt(ee) / torch.sqrt(ss)


def corr(preds, labels):
    # preds = preds.view(-1, 1)
    # labels = labels.view(-1, 1)

    lm = labels - torch.mean(labels)
    pm = preds - torch.mean(preds)

    xy = torch.mm(lm.transpose(1, 0), pm)
    xx = torch.mm(lm.transpose(1, 0), lm)
    yy = torch.mm(pm.transpose(1, 0), pm)

    return (xy / torch.sqrt(xx * yy)).view(-1)
