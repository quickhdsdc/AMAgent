from __future__ import annotations

import torch
import torch.nn.functional as F

from .task_labels import NUM_LABELS

NONE_IDX = 0
LOF_IDX = 1
BALLING_IDX = 2
KEYHOLE_IDX = 3

GOOD_IDX = 0
DEFECTIVE_IDX = 1


VED_LOF_UPPER = 50.0
VED_CONDUCTION_UPPER = 100.0
VED_TRANSITION_UPPER = 120.0


def compute_ved(
    power_w: torch.Tensor,
    velocity_mm_s: torch.Tensor,
    hatch_um: torch.Tensor,
    layer_um: torch.Tensor,
) -> torch.Tensor:
    """Vectorised VED = P / (v * h_mm * t_mm).

    Returns NaN for rows where any of v / h / t is zero or non-finite.
    Caller is expected to mask invalid rows out before averaging the
    physics loss.
    """

    hatch_mm = hatch_um / 1000.0
    layer_mm = layer_um / 1000.0

    denom = velocity_mm_s * hatch_mm * layer_mm
    safe_denom = torch.where(denom == 0, torch.full_like(denom, float("nan")), denom)
    return power_w / safe_denom


def build_forbidden_mask(ved: torch.Tensor, num_labels: int = NUM_LABELS) -> torch.Tensor:
    """Construct a (B, num_labels) 0/1 mask of *forbidden* labels per row.

    `ved` is shape (B,). Rows with NaN VED get an all-zero mask (no
    constraint applied) — those rows will be removed from the loss via
    `valid_mask` upstream.

    The mask depends on the active task:
      * 4-class {none, lof, balling, keyhole} — VED windows forbid the
        physically-implausible defect modes.
      * binary {good, defective} — the stable conduction window forbids
        `defective`; the extreme (LoF or keyhole) regimes forbid `good`.
        Mapping per `readme_exp.md` §4.
    """

    device = ved.device
    dtype = torch.float32
    batch_size = ved.size(0)
    mask = torch.zeros(batch_size, num_labels, device=device, dtype=dtype)

    if num_labels == 2:
        stable = (ved >= VED_LOF_UPPER) & (ved <= VED_CONDUCTION_UPPER)
        mask[stable, DEFECTIVE_IDX] = 1.0
        extreme = (ved < VED_LOF_UPPER) | (ved > VED_TRANSITION_UPPER)
        mask[extreme, GOOD_IDX] = 1.0
        nan_rows = torch.isnan(ved)
        if nan_rows.any():
            mask[nan_rows] = 0.0
        return mask

    keyhole_regime = ved > VED_TRANSITION_UPPER
    mask[keyhole_regime, NONE_IDX] = 1.0
    mask[keyhole_regime, LOF_IDX] = 1.0
    mask[keyhole_regime, BALLING_IDX] = 1.0

    conduction_regime = (ved >= VED_LOF_UPPER) & (ved <= VED_CONDUCTION_UPPER)
    mask[conduction_regime, LOF_IDX] = 1.0
    mask[conduction_regime, BALLING_IDX] = 1.0
    mask[conduction_regime, KEYHOLE_IDX] = 1.0

    lof_regime = ved < VED_LOF_UPPER
    mask[lof_regime, KEYHOLE_IDX] = 1.0


    nan_rows = torch.isnan(ved)
    if nan_rows.any():
        mask[nan_rows] = 0.0
    return mask


def physics_loss_from_logits(
    logits: torch.Tensor,
    raw_params: torch.Tensor,
    num_labels: int = NUM_LABELS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean physics-violation loss over a batch.

    Parameters
    ----------
    logits : (B, num_labels) raw classifier logits.
    raw_params : (B, 4) tensor with columns [P_w, v_mm_s, h_um, t_um].

    Returns
    -------
    phys_loss : scalar tensor, the mean over *valid* rows of the
        sum of probabilities on forbidden labels.
    valid_count : scalar tensor giving the number of valid rows in
        the batch (for logging only).
    """

    probs = F.softmax(logits, dim=-1)

    power = raw_params[:, 0]
    velocity = raw_params[:, 1]
    hatch = raw_params[:, 2]
    layer = raw_params[:, 3]

    ved = compute_ved(power, velocity, hatch, layer)
    forbidden_mask = build_forbidden_mask(ved, num_labels=num_labels)

    per_row = (probs * forbidden_mask).sum(dim=-1)

    valid = torch.isfinite(ved)
    valid_count = valid.sum()

    if valid_count.item() == 0:
        return per_row.sum() * 0.0, valid_count

    masked = per_row * valid.float()
    phys_loss = masked.sum() / valid_count.float()
    return phys_loss, valid_count


def physics_violation_rate(
    pred_labels: torch.Tensor,
    raw_params: torch.Tensor,
    num_labels: int = NUM_LABELS,
) -> tuple[float, int]:
    """Fraction of predictions falling into the VED-forbidden set.

    Used at evaluation time to verify that the physics-constraint loss
    actually reduces test-time violations vs the C7 baseline.

    Returns (rate, n_valid) — rate is NaN if no valid rows.
    """

    power = raw_params[:, 0]
    velocity = raw_params[:, 1]
    hatch = raw_params[:, 2]
    layer = raw_params[:, 3]

    ved = compute_ved(power, velocity, hatch, layer)
    forbidden_mask = build_forbidden_mask(ved, num_labels=num_labels)

    valid = torch.isfinite(ved)
    if valid.sum().item() == 0:
        return float("nan"), 0

    pred_one_hot = F.one_hot(pred_labels.long(), num_classes=num_labels).float()
    violated = (pred_one_hot * forbidden_mask).sum(dim=-1) > 0
    n_valid = int(valid.sum().item())
    rate = float((violated & valid).sum().item()) / n_valid
    return rate, n_valid
