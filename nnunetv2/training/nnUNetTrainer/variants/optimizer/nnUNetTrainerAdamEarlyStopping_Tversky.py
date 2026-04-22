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
from nnunetv2.training.loss.compound_losses import Tversky_and_CE_loss, Tversky_and_CE_loss_PerClass
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.tversky import MemoryEfficientSoftTverskyLoss, PerClassFocalTverskyLoss


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
        # Frequencies measured on 500 random TRAINING TILES (actual tile distribution):
        #   Class 0 Background           57.51 %   w = 0.1925
        #   Class 1 ConnectivePerineurium 23.32 %   w = 0.4746
        #   Class 2 Adipose               6.37 %   w = 1.7389  ← minority in tiles
        #   Class 3 NerveFascicle        12.80 %   w = 0.8647
        #   Class 4 Blood_vessel          0.00 %   w = 1.7294  (capped at 2×w3)
        # Weights normalised so that all 5 sum to 5.
        ce_weights = torch.tensor(
            [0.1925, 0.4746, 1.7389, 0.8647, 1.7294],
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


class nnUNetTrainerAdamEarlyStopping_TverskyPerClass(nnUNetTrainerAdamEarlyStopping):
    """
    Per-class Focal Tversky + corrected weighted CE trainer.

    Improvements over the uniform Tversky trainer:
      1. Per-class alpha/beta: adipose (cls 2) and fascicle (cls 3) get higher
         recall bias (beta=0.8) while background/connective stay balanced (beta=0.6).
      2. Focal exponent gamma=1.33: down-weights easy well-segmented regions and
         amplifies gradients for hard/missed adipose and fascicle pixels.
      3. Per-class Tversky loss weights (1.5x for adipose/fascicle vs 0.5x for bg).
      4. CE weights from actual 500-tile distribution (adipose w=1.74, not 0.77).

    See documentation/improvement_plan_recall.md for full rationale.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        self.initial_lr = 3e-4
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_optimizers(self):
        self.initial_lr = 3e-4
        return super().configure_optimizers()

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()

        from nnunetv2.utilities.helpers import softmax_helper_dim1

        # ── Per-class α, β (сlasses 1–4, background excluded via do_bg=False) ───
        # Higher recall bias (β=0.8) for adipose (pos 1 in tensor = cls 2)
        # and fascicle (pos 2 = cls 3); balanced for connective and blood vessel.
        alpha_per_class = [0.4, 0.2, 0.2, 0.4]   # cls 1, 2, 3, 4
        beta_per_class  = [0.6, 0.8, 0.8, 0.6]   # cls 1, 2, 3, 4
        tversky_cw      = [0.5, 1.5, 1.5, 0.5]   # amplify adipose + fascicle

        # ── CE weights from actual training-tile distribution ────────────────
        ce_weights = torch.tensor(
            [0.1925, 0.4746, 1.7389, 0.8647, 1.7294],
            dtype=torch.float32, device=self.device,
        )

        loss = Tversky_and_CE_loss_PerClass(
            tversky_kwargs=dict(
                apply_nonlin=softmax_helper_dim1,
                alpha=alpha_per_class,
                beta=beta_per_class,
                class_weights=tversky_cw,
                gamma=1.33,
                smooth=1.0,
                do_bg=False,
                batch_dice=self.configuration_manager.batch_dice,
                ddp=self.is_ddp,
            ),
            ce_kwargs={'weight': ce_weights},
            weight_ce=1.0,
            weight_tversky=1.0,
            ignore_label=self.label_manager.ignore_label,
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


class nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft(nnUNetTrainerAdamEarlyStopping):
    """
    Softened Per-class Focal Tversky + CE trainer.

    A milder re-tuning of TverskyPerClass after it was found to degrade
    NerveFascicle (Dice 0.914 → 0.822) and Blood vessel (0.626 → 0.420)
    while only partially recovering Adipose over the TV_Nyul baseline.

    Root cause: high β + high class_weights + high γ over-penalised FNs on
    hard/domain-shifted classes and pushed the optimiser to inflate background.
    The softened params keep a meaningful recall bias without destabilising the
    other foreground classes.

    Changes vs TverskyPerClass (aggressive):
      alpha:         [0.4, 0.2, 0.2, 0.4]  →  [0.3, 0.3, 0.3, 0.3]
      beta:          [0.6, 0.8, 0.8, 0.6]  →  [0.70, 0.75, 0.75, 0.80]
      class_weights: [0.5, 1.5, 1.5, 0.5]  →  [1.0, 1.2, 1.2, 1.5]
      gamma:          1.33                 →   1.15

    Blood vessel kept at the highest β (0.80) and weight (1.5) because it is the
    rarest class (0–1.4 % GT) and was most damaged by the aggressive version.
    Adipose and fascicle get a modest recall bias without fully overwhelming
    connective tissue gradients.

    Best used alongside BioAegisOversample (sampling fix) rather than as a
    standalone loss fix — the Adipose gap on Bio-Aegis is primarily a domain-shift
    / underrepresentation problem, not solely a loss problem.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        self.initial_lr = 3e-4
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_optimizers(self):
        self.initial_lr = 3e-4
        return super().configure_optimizers()

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()

        from nnunetv2.utilities.helpers import softmax_helper_dim1

        # ── Per-class α, β (foreground classes 1–4, background excluded) ──────
        # Uniform FP penalty; graduated recall bias with blood vessel highest.
        alpha_per_class = [0.3,  0.3,  0.3,  0.3 ]   # cls 1 (connec), 2 (adip), 3 (nerve), 4 (blood)
        beta_per_class  = [0.70, 0.75, 0.75, 0.80]   # cls 1, 2, 3, 4
        tversky_cw      = [1.0,  1.2,  1.2,  1.5 ]   # blood highest; mild boost for adip/nerve

        # ── CE weights from actual training-tile distribution ────────────────
        ce_weights = torch.tensor(
            [0.1925, 0.4746, 1.7389, 0.8647, 1.7294],
            dtype=torch.float32, device=self.device,
        )

        loss = Tversky_and_CE_loss_PerClass(
            tversky_kwargs=dict(
                apply_nonlin=softmax_helper_dim1,
                alpha=alpha_per_class,
                beta=beta_per_class,
                class_weights=tversky_cw,
                gamma=1.15,
                smooth=1.0,
                do_bg=False,
                batch_dice=self.configuration_manager.batch_dice,
                ddp=self.is_ddp,
            ),
            ce_kwargs={'weight': ce_weights},
            weight_ce=1.0,
            weight_tversky=1.0,
            ignore_label=self.label_manager.ignore_label,
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
