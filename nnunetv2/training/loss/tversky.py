"""
Tversky Loss for improved recall on minority classes (e.g., fascicles).

Tversky Index: TP / (TP + α*FP + β*FN)
- α controls penalty for false positives
- β controls penalty for false negatives
- When α=β=0.5, equivalent to Dice loss
- When β > α, model is penalized more for missing true positives (improves recall)

Recommended: α=0.3, β=0.7 for better fascicle detection
"""
from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch import nn


class SoftTverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7):
        """
        Soft Tversky Loss.
        
        Args:
            apply_nonlin: Nonlinearity to apply (e.g., softmax)
            batch_dice: Whether to compute Tversky over the batch dimension
            do_bg: Whether to include background class
            smooth: Smoothing factor to avoid division by zero
            ddp: Whether using distributed data parallel
            alpha: Weight for false positives (default 0.3)
            beta: Weight for false negatives (default 0.7) - higher = more recall
        """
        super(SoftTverskyLoss, self).__init__()
        
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0, dtype=torch.float32)
            fp = AllGatherGrad.apply(fp).sum(0, dtype=torch.float32)
            fn = AllGatherGrad.apply(fn).sum(0, dtype=torch.float32)

        # Tversky index: TP / (TP + α*FP + β*FN)
        nominator = tp
        fp_clamped = fp.clamp_min(0)  # Clamp to avoid negative due to float precision
        fn_clamped = fn.clamp_min(0)  # Clamp to avoid negative due to float precision
        denominator = tp + self.alpha * fp_clamped + self.beta * fn_clamped

        tversky = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        
        tversky = tversky.mean()

        return -tversky


class MemoryEfficientSoftTverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7):
        """
        Memory-efficient Soft Tversky Loss (analogous to MemoryEfficientSoftDiceLoss).
        
        Args:
            apply_nonlin: Nonlinearity to apply (e.g., softmax)
            batch_dice: Whether to compute Tversky over the batch dimension
            do_bg: Whether to include background class
            smooth: Smoothing factor to avoid division by zero
            ddp: Whether using distributed data parallel
            alpha: Weight for false positives (default 0.3)
            beta: Weight for false negatives (default 0.7) - higher = more recall
        """
        super(MemoryEfficientSoftTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)

            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        # TP = intersect
        # FP = sum_pred - intersect
        # FN = sum_gt - intersect
        # Tversky = TP / (TP + α*FP + β*FN)
        tp = intersect
        fp = (sum_pred - intersect).clamp_min(0)  # Clamp to avoid negative due to float precision
        fn = (sum_gt - intersect).clamp_min(0)    # Clamp to avoid negative due to float precision
        
        denominator = tp + self.alpha * fp + self.beta * fn + float(self.smooth)
        tversky = (tp + self.smooth) / denominator.clamp_min(1e-8)

        tversky = tversky.mean()
        return -tversky


class PerClassFocalTverskyLoss(nn.Module):
    """
    Per-class Focal Tversky Loss.

    Each class has its own alpha (FP penalty) and beta (FN penalty), allowing
    recall-biased settings for minority or hard classes (adipose, fascicle) while
    keeping more balanced settings for easy/dominant classes (background).

    A focal exponent gamma >= 1 down-weights easy well-segmented regions and
    amplifies the gradient from hard misclassified pixels (analogous to focal CE).

    Loss per class c (background excluded when do_bg=False):
        tversky_c    = TP_c / (TP_c + alpha_c * FP_c + beta_c * FN_c + smooth)
        focal_loss_c = (1 - tversky_c) ^ gamma
        loss         = sum(class_weights_c / sum(class_weights) * focal_loss_c)

    Recommended for nerve segmentation (5 classes, do_bg=False → 4 values):
        alpha         = [0.4, 0.2, 0.2, 0.4]   # cls 1,2,3,4
        beta          = [0.6, 0.8, 0.8, 0.6]   # cls 1,2,3,4  — high recall for adipose+fascicle
        class_weights = [0.5, 1.5, 1.5, 0.5]   # emphasise adipose (cls2) + fascicle (cls3)
        gamma         = 1.33
    """

    def __init__(
        self,
        apply_nonlin: Callable = None,
        alpha=None,           # per-class FP penalty; scalar, list, or Tensor [C]
        beta=None,            # per-class FN penalty; scalar, list, or Tensor [C]
        class_weights=None,   # per-class aggregation weights; None = uniform
        gamma: float = 1.33,  # focal exponent; 1.0 = standard Tversky (no focal)
        smooth: float = 1.0,
        do_bg: bool = False,
        batch_dice: bool = False,
        ddp: bool = True,
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.gamma = gamma
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.ddp = ddp
        # Store initialisation values; tensors are built lazily on first forward
        self._alpha_init = alpha
        self._beta_init = beta
        self._cw_init = class_weights

    @staticmethod
    def _to_tensor(value, C: int, device, dtype):
        """Convert scalar / list / Tensor to a [C] tensor."""
        if value is None:
            return torch.ones(C, dtype=dtype, device=device)
        if isinstance(value, (int, float)):
            return torch.full((C,), float(value), dtype=dtype, device=device)
        t = torch.as_tensor(value, dtype=dtype, device=device)
        return t.expand(C) if t.numel() == 1 else t

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))   # spatial dims only

        # One-hot encode ground truth
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.float()
            else:
                y_onehot = torch.zeros_like(x)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = (
                y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            )

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred  = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred  = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred  = AllGatherGrad.apply(sum_pred).sum(0,  dtype=torch.float32)
                sum_gt    = AllGatherGrad.apply(sum_gt).sum(0,    dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred  = sum_pred.sum(0,  dtype=torch.float32)
            sum_gt    = sum_gt.sum(0,    dtype=torch.float32)
        else:
            intersect = intersect.mean(0)
            sum_pred  = sum_pred.mean(0)
            sum_gt    = sum_gt.mean(0)

        C = intersect.shape[0]
        dtype, device = intersect.dtype, intersect.device
        alpha = self._to_tensor(self._alpha_init, C, device, dtype)
        beta  = self._to_tensor(self._beta_init,  C, device, dtype)
        cw    = self._to_tensor(self._cw_init,    C, device, dtype)
        cw    = cw / cw.sum()   # normalise so weights sum to 1

        tp = intersect
        fp = (sum_pred - intersect).clamp_min(0)
        fn = (sum_gt   - intersect).clamp_min(0)

        denominator = (tp + alpha * fp + beta * fn + self.smooth).clamp_min(1e-8)
        tversky_c   = (tp + self.smooth) / denominator   # [C]
        focal_c     = (1.0 - tversky_c).pow(self.gamma)  # [C]
        return (focal_c * cw).sum()                       # scalar (positive, to be minimised)


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    tl_standard = SoftTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False, alpha=0.3, beta=0.7)
    tl_efficient = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False, alpha=0.3, beta=0.7)
    
    res_standard = tl_standard(pred, ref)
    res_efficient = tl_efficient(pred, ref)
    print(f"Standard Tversky: {res_standard}, Memory-efficient Tversky: {res_efficient}")
