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


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    tl_standard = SoftTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False, alpha=0.3, beta=0.7)
    tl_efficient = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False, alpha=0.3, beta=0.7)
    
    res_standard = tl_standard(pred, ref)
    res_efficient = tl_efficient(pred, ref)
    print(f"Standard Tversky: {res_standard}, Memory-efficient Tversky: {res_efficient}")
