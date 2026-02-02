import torch


def nannorm(unormalized_tensor, keepdim=False, dim=None, eps=1e-6):
    return torch.nansum(torch.abs(unormalized_tensor) ** 2, dim=dim, keepdim=keepdim).sqrt().clamp_min(eps)


def nanstd(x, dim, keepdim=False, eps=1e-6):
    mask = ~torch.isnan(x)
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1)
    x0 = torch.where(mask, x, 0.0)
    mean = x0.sum(dim=dim, keepdim=True) / count
    var = torch.where(mask, (x - mean) ** 2, 0.0).sum(dim=dim, keepdim=True) / count
    std = var.sqrt().clamp_min(eps)
    if keepdim:
        return std
    if isinstance(dim, tuple):
        for d in sorted(dim, reverse=True):
            std = std.squeeze(d)
        return std
    return std.squeeze(dim)


def log_nanmean_exp(x, dim=None):
    """Log of mean of exp, ignoring NaNs. Numerically stable."""
    if dim is None:
        x = x.flatten()
        dim = 0
    # Replace NaNs with -inf so exp(-inf) = 0
    x_clean = torch.where(torch.isnan(x), torch.full_like(x, -float("inf")), x)
    # Count non-NaN elements
    n = (~torch.isnan(x)).sum(dim=dim).clamp_min(1)
    return torch.logsumexp(x_clean, dim=dim) - torch.log(n.float())


def corr_loss(y_true, y_hat, dim=None, eps=1e-6):
    # Create mask of positions valid in BOTH tensors
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y_true)).float()
    valid_count = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1)

    # Replace NaN with 0 to prevent NaN gradient propagation (mask handles statistics)
    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

    # Compute means over valid positions only
    y_hat_mean = (y_hat * valid_mask).sum(dim=dim, keepdim=True) / valid_count
    y_true_mean = (y_true * valid_mask).sum(dim=dim, keepdim=True) / valid_count

    # Center the data (mask zeros out invalid positions)
    y_hat_centered = (y_hat - y_hat_mean) * valid_mask
    y_true_centered = (y_true - y_true_mean) * valid_mask

    # Compute std over valid positions
    y_hat_std = (y_hat_centered.pow(2).sum(dim=dim, keepdim=True) / valid_count).sqrt().clamp_min(eps)
    y_true_std = (y_true_centered.pow(2).sum(dim=dim, keepdim=True) / valid_count).sqrt().clamp_min(eps)

    # Normalize
    y_hat_norm = y_hat_centered / y_hat_std
    y_true_norm = y_true_centered / y_true_std

    # Compute correlation over valid positions
    corr = (y_hat_norm * y_true_norm).sum(dim=dim, keepdim=True) / valid_count

    ave_corr = torch.mean(corr)

    return 1.0 - ave_corr


def cos_loss(y_true, y_hat, dim=None, eps=1e-6):
    # Create mask of positions valid in BOTH tensors
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y_true)).float()

    # Replace NaN with 0 to prevent NaN gradient propagation (mask handles statistics)
    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

    # Apply mask so norms are computed only over valid positions
    y_hat_masked = y_hat * valid_mask
    y_true_masked = y_true * valid_mask

    # Normalize (nannorm uses nansum internally, but all values are now valid)
    y_hat_norm = y_hat_masked / nannorm(y_hat_masked, dim=dim, keepdim=True).clamp_min(eps)
    y_true_norm = y_true_masked / nannorm(y_true_masked, dim=dim, keepdim=True).clamp_min(eps)

    # Cosine similarity over valid positions
    cosine_sim = torch.sum(y_hat_norm * y_true_norm, dim=dim, keepdim=True)

    ave_cosine_sim = torch.mean(cosine_sim)

    return 1.0 - ave_cosine_sim


def mse_loss(y_true, y_hat, dim=None, eps=1e-6):
    # Create mask of positions valid in BOTH tensors
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y_true)).float()

    # Replace NaN with 0 to prevent NaN gradient propagation (mask handles statistics)
    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

    diff2 = (y_hat - y_true).pow(2) * valid_mask
    valid_count = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1)
    mse = diff2.sum(dim=dim, keepdim=True) / valid_count
    return torch.mean(mse)


def combined_loss(y_true, y_hat, eps=1e-6):
    cos_loss_val = cos_loss(y_true, y_hat, eps=eps)
    corr_loss_val = corr_loss(y_true, y_hat, eps=eps)
    return 0.5 * cos_loss_val + 0.5 * corr_loss_val


def calc_loss_dict(loss_fn, pred, target):
    keys = list(pred.keys())
    num_readouts = len(keys)

    loss = loss_fn(target[keys[0]], pred[keys[0]]) / num_readouts
    for i in keys[1:]:
        loss = loss + loss_fn(target[i], pred[i]) / num_readouts

    return loss
