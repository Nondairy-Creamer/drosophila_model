import torch
import neural_models as nm
import readout_utilities as ru


# ---------------------------------------------------------------------------
# Primitive helpers (NaN-safe)
# ---------------------------------------------------------------------------

def nannorm(unormalized_tensor, keepdim=False, dim=None, eps=1e-6):
    return torch.nansum(torch.abs(unormalized_tensor) ** 2, dim=dim, keepdim=keepdim).sqrt().clamp_min(eps)


# ---------------------------------------------------------------------------
# Canonical loss primitives
# ---------------------------------------------------------------------------

def normalized_mse(pred, target, eps=1e-8):
    """MSE divided by mean-square of target (normalized to ~0-1 range).

    NaN entries in *target* (and corresponding *pred*) are excluded.
    """
    valid = ~torch.isnan(target) & ~torch.isnan(pred)
    if not valid.any():
        return pred.new_tensor(0.0)
    p, t = pred[valid], target[valid]
    return (p - t).pow(2).mean() / (t.pow(2).mean() + eps)


def cosine_loss(pred, target, eps=1e-8):
    """Cosine distance over flattened valid entries.

    Treats the full tensor as a single vector after masking out NaN and
    near-zero-magnitude rows (when the last dim is >1).
    """
    pred_flat = pred.reshape(-1, pred.shape[-1]) if pred.dim() > 1 else pred.reshape(-1, 1)
    target_flat = target.reshape(-1, target.shape[-1]) if target.dim() > 1 else target.reshape(-1, 1)
    has_nan = torch.isnan(target_flat).any(dim=-1) | torch.isnan(pred_flat).any(dim=-1)
    mag = target_flat.norm(dim=-1)
    valid = (mag > eps) & ~has_nan
    if not valid.any():
        return pred.new_tensor(0.0)
    p = pred_flat[valid].reshape(-1)
    t = target_flat[valid].reshape(-1)
    cos_sim = torch.dot(p, t) / (p.norm() * t.norm() + eps)
    return 1.0 - cos_sim


# ---------------------------------------------------------------------------
# Evaluation helpers (used by analysis scripts)
# ---------------------------------------------------------------------------

def cos_loss(y_true, y_hat, dim=None, eps=1e-6):
    """NaN-safe cosine distance: ``1 - cosine_similarity``."""
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y_true)).float()

    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

    y_hat_masked = y_hat * valid_mask
    y_true_masked = y_true * valid_mask

    y_hat_norm = y_hat_masked / nannorm(y_hat_masked, dim=dim, keepdim=True).clamp_min(eps)
    y_true_norm = y_true_masked / nannorm(y_true_masked, dim=dim, keepdim=True).clamp_min(eps)

    cosine_sim = torch.sum(y_hat_norm * y_true_norm, dim=dim, keepdim=True)
    return 1.0 - torch.mean(cosine_sim)


def mse_loss(y_true, y_hat, dim=None, eps=1e-6):
    """NaN-safe mean squared error."""
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y_true)).float()

    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

    diff2 = (y_hat - y_true).pow(2) * valid_mask
    valid_count = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1)
    mse = diff2.sum(dim=dim, keepdim=True) / valid_count
    return torch.mean(mse)


# ---------------------------------------------------------------------------
# Training loss components
#
# Unified interface:
#   component_fn(model, outputs, target, train_mask, loss_start_step) -> scalar
#
# ``outputs`` is the dict produced by predict (readout keys + 'voltage'/
# 'firing_rate' + optionally 'pred_rotation'/'pred_translation').
# ``target`` is a dict with possible keys 'optic_flow', 'rotation',
# 'translation'.
# ---------------------------------------------------------------------------

def _mask_pred_target(pred, target_flow, train_mask, loss_start_step):
    """Apply time cropping and spatial masking, returning masked pred/target."""
    pred_t = pred[:, loss_start_step:]
    target_t = target_flow[:, loss_start_step:]
    mask_t = train_mask[:, loss_start_step:]

    expand_shape = pred_t.shape
    invalid = ~mask_t.unsqueeze(-1).expand(expand_shape)
    pred_t = pred_t.clone()
    target_t = target_t.clone()
    pred_t[invalid] = float('nan')
    target_t[invalid] = float('nan')
    return pred_t, target_t


def flow_loss(model, outputs, target, train_mask, loss_start_step):
    """Normalized MSE between predicted flow and ground-truth optic flow.

    Averages across all flow-type readouts present in *outputs*.
    """
    target_flow = target['optic_flow']
    losses = []
    for key, mod in model._readout_modules.items():
        if not mod.eye_asymmetric or key not in outputs:
            continue
        pred_t, target_t = _mask_pred_target(outputs[key], target_flow,
                                             train_mask, loss_start_step)
        losses.append(normalized_mse(pred_t, target_t))
    if not losses:
        return target_flow.new_tensor(0.0)
    return sum(losses) / len(losses)


def odometry_loss(model, outputs, target, train_mask, loss_start_step):
    """Rotation MSE + translation cosine + integrated position terms."""
    pred_rot = outputs['pred_rotation'][:, loss_start_step:]
    pred_trans = outputs['pred_translation'][:, loss_start_step:]
    target_rot = target['rotation'][:, loss_start_step:]
    target_trans = target['translation'][:, loss_start_step:]

    loss_rot_rate = normalized_mse(pred_rot, target_rot)
    loss_trans_rate = cosine_loss(pred_trans, target_trans)

    pred_rot_pos = nm.integrate_angular_velocity(pred_rot, model.dt)
    target_rot_pos = nm.integrate_angular_velocity(target_rot, model.dt)
    loss_rot_pos = normalized_mse(pred_rot_pos, target_rot_pos)

    pred_trans_pos = nm.integrate_translational_velocity(pred_trans, pred_rot, model.dt)
    target_trans_pos = nm.integrate_translational_velocity(target_trans, target_rot, model.dt)
    loss_trans_pos = cosine_loss(pred_trans_pos, target_trans_pos)

    return (loss_rot_rate + loss_trans_rate + loss_rot_pos + loss_trans_pos) / 4


def l1_reg(model, outputs, target, train_mask, loss_start_step):
    """L1 on input/output strength parameters and any readout conv weights."""
    terms = [
        model.input_strength_log.exp().mean(),
        model.output_strength_log.exp().mean(),
    ]
    for mod in model._readout_modules.values():
        for sub in mod.modules():
            if hasattr(sub, 'conv_layers'):
                terms.append(
                    torch.stack([p.abs().mean() for p in sub.conv_layers.parameters()]).mean()
                )
                break
    return sum(terms) / len(terms)


def l2_reg(model, outputs, target, train_mask, loss_start_step):
    """L2 on input/output strength parameters."""
    return (model.input_strength_log.exp().pow(2).mean()
            + model.output_strength_log.exp().pow(2).mean()) / 2


def sigma_reg(model, outputs, target, train_mask, loss_start_step):
    """Penalty when spectral radius exceeds the allowed maximum."""
    sigma_max = model.calc_sigma_max(max_iter=10)
    return torch.relu(sigma_max - model.max_allowed_sigma) ** 2


def motion_mag(model, outputs, target, train_mask, loss_start_step):
    """Normalized MSE between predicted motion magnitude and ||optic_flow||."""
    target_flow = target['optic_flow']
    target_mag = target_flow.norm(dim=-1)           # (batch, time, cols)
    mag_key = next(
        (k for k, m in model._readout_modules.items()
         if isinstance(m, ru.MotionMagReadout)),
        None,
    )
    if mag_key is None:
        return target_flow.new_tensor(0.0)
    pred_mag = outputs[mag_key]
    pred_t, target_t = _mask_pred_target(
        pred_mag, target_mag, train_mask, loss_start_step)
    return normalized_mse(pred_t, target_t)

