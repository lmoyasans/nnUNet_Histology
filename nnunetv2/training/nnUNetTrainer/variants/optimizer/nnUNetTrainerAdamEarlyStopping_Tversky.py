"""
nnUNet Trainer with Tversky Loss for improved recall on minority classes.

Based on nnUNetTrainerAdamEarlyStopping with:
- Tversky loss instead of Dice loss (α=0.3, β=0.7 for better recall)
- Adam optimizer with early stopping
- Gradient clipping and NaN protection

Tversky Index: TP / (TP + α*FP + β*FN)
- α=0.3: Lower penalty for false positives
- β=0.7: Higher penalty for false negatives (forces better recall)

Use this for datasets where detecting all instances of a class is more
important than avoiding false positives (e.g., nerve fascicle detection).
"""
import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamEarlyStopping import nnUNetTrainerAdamEarlyStopping
from nnunetv2.training.loss.compound_losses import Tversky_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.tversky import MemoryEfficientSoftTverskyLoss


class nnUNetTrainerAdamEarlyStopping_Tversky(nnUNetTrainerAdamEarlyStopping):
    """
    Adam trainer with Tversky Loss and Early Stopping.
    
    Uses Tversky loss (α=0.3, β=0.7) for improved recall on minority classes.
    Inherits all early stopping, NaN protection, and gradient clipping from parent.
    
    Uses lower learning rate (3e-4) than default (1e-2) for stability with Tversky loss.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Tversky parameters - can be overridden in subclasses
        self.tversky_alpha = 0.3  # FP penalty (lower = allow more FP)
        self.tversky_beta = 0.7   # FN penalty (higher = better recall)
        
        # Lower LR MUST be set BEFORE super().__init__() because configure_optimizers
        # is called during initialization
        self.initial_lr = 3e-4
        
        super().__init__(plans, configuration, fold, dataset_json, device)
    
    def configure_optimizers(self):
        """Configure optimizer with lower LR for Tversky loss stability."""
        self.initial_lr = 3e-4  # Ensure LR is set before optimizer creation
        return super().configure_optimizers()

    def _build_loss(self):
        """
        Build Tversky + CE loss with class weights.
        
        Tversky loss with β > α penalizes false negatives more, improving recall.
        Combined with weighted CE where class weights prioritize minority classes.
        """
        if self.label_manager.has_regions:
            # Region-based training: fall back to default
            return super()._build_loss()

        # Inverse-frequency weights, normalised to sum to num_classes (5).
        # Same methodology as nnUNetTrainerAdamEarlyStopping._build_loss.
        # Frequencies measured on the full 5-class combined dataset (300-case sample):
        #   Class 0 Background           40.06 %   w = 0.384
        #   Class 1 ConnectivePerineurium 25.79 %   w = 0.597
        #   Class 2 Adipose              19.93 %   w = 0.772
        #   Class 3 NerveFascicle        14.22 %   w = 1.082
        #   Class 4 Blood_vessel          0.005%   w = 2.164  (capped at 2×w3 to avoid instability)
        # Weights are normalised so that all 5 sum to 5.
        ce_weights = torch.tensor(
            [0.3840, 0.5969, 0.7722, 1.0819, 2.1639],
            dtype=torch.float32,
            device=self.device,
        )

        loss = Tversky_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1.0, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': ce_weights},
            weight_ce=1.0, weight_tversky=1.0,
            ignore_label=self.label_manager.ignore_label,
            tversky_class=MemoryEfficientSoftTverskyLoss,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
        )

        if self._do_i_compile():
            loss.tversky = torch.compile(loss.tversky)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerAdamEarlyStopping_TverskyHighRecall(nnUNetTrainerAdamEarlyStopping_Tversky):
    """
    Even more aggressive recall optimization: α=0.2, β=0.8
    Use when false negatives are very costly.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.tversky_alpha = 0.2
        self.tversky_beta = 0.8


class nnUNetTrainerAdamEarlyStopping_TverskyBalanced(nnUNetTrainerAdamEarlyStopping_Tversky):
    """
    Balanced Tversky: α=0.4, β=0.6
    Slight recall bias but more balanced than default.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.tversky_alpha = 0.4
        self.tversky_beta = 0.6
