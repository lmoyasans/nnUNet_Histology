import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdam import nnUNetTrainerAdam
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class nnUNetTrainerAdamEarlyStopping(nnUNetTrainerAdam):
    """
    Adam trainer with early stopping based on validation loss.

    Stops training if validation loss doesn't improve for `patience` epochs.
    Includes protection against NaN losses and gradient clipping for stability.

    NOTE: nnUNet val losses are negative (Dice-based), so "improvement" means
    the value becomes *more negative*.  All spike/improvement logic uses the
    signed difference rather than a ratio so it works correctly with negative
    values.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Initialize stability parameters BEFORE parent class
        self.patience = 50           # epochs to wait for improvement
        self.min_delta = 1e-4        # minimum improvement to count
        self.max_grad_norm = 1.0     # gradient clipping max norm
        self.max_nan_tolerance = 3   # stop after this many NaN occurrences
        # Absolute margin: stop if loss worsens by more than this vs best
        # (works for both positive and negative losses)
        self.max_loss_spike_margin = 0.5

        # Early stopping state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.nan_count = 0

        super().__init__(plans, configuration, fold, dataset_json, device)

    def _build_loss(self):
        """
        Override the default loss to add inverse-frequency class weights to the
        cross-entropy component.

        Class pixel frequencies measured over all 2,264 training images:
          Class 0 Background           47.81 %   → weight 0.30
          Class 1 ConnectivePerineurium 29.54 %   → weight 0.49
          Class 2 Adipose               6.14 %   → weight 2.34  ← minority / hard
          Class 3 NerveFascicle        16.51 %   → weight 0.87

        Weights are inverse-frequency, normalised to sum to num_classes (4),
        so the total gradient magnitude stays comparable to uniform weighting.
        Dice loss runs unweighted (it is already class-balanced by design).
        """
        if self.label_manager.has_regions:
            # Region-based training: fall back to default (no class weights)
            return super()._build_loss()

        # Inverse-frequency weights, normalised to sum to 4
        # Recompute with:  w_i = (1 / freq_i) / sum(1/freq_j) * num_classes
        ce_weights = torch.tensor(
            [0.3007, 0.4871, 2.3416, 0.8710],
            dtype=torch.float32,
            device=self.device,
        )

        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': ce_weights},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

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

    def train_step(self, batch: dict) -> dict:
        """
        Override train step with:
        - Correct autocast usage (never torch.no_grad during training)
        - NaN loss detection before backward (avoids poisoning parameters)
        - Gradient clipping for stability
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)


        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if torch.isnan(l) or torch.isinf(l):
            self.print_to_log_file(
                f"[train_step] NaN/Inf loss detected — skipping backward for this batch."
            )
            return {'loss': np.array(np.nan)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}
        
    def on_validation_epoch_end(self, val_outputs: list):
        """
        Called after each validation epoch.
        Checks if we should stop training based on validation loss.
        Includes NaN detection and loss spike protection.

        NOTE: nnUNet val losses are negative (Dice-CE).  Spike detection uses
        an absolute margin rather than a ratio to stay correct for negative values.
        """
        super().on_validation_epoch_end(val_outputs)

        current_val_loss = self.logger.get_value('val_losses', step=-1)

        # --- NaN / Inf guard ------------------------------------------------
        if np.isnan(current_val_loss) or np.isinf(current_val_loss):
            self.nan_count += 1
            self.print_to_log_file(
                f"\n{'!'*50}\n"
                f"WARNING: NaN or Inf loss detected! "
                f"(occurrence {self.nan_count}/{self.max_nan_tolerance})\n"
                f"Current validation loss: {current_val_loss}\n"
                f"{'!'*50}\n"
            )
            if self.nan_count >= self.max_nan_tolerance:
                self.should_stop = True
                self.print_to_log_file(
                    f"\n{'='*50}\n"
                    f"Stopping training: Too many NaN/Inf losses ({self.nan_count})!\n"
                    f"Best validation loss: {self.best_val_loss:.6f}\n"
                    f"{'='*50}\n"
                )
            return

        # --- Loss spike guard -----------------------------------------------
        if self.best_val_loss != float('inf'):
            loss_worsening = current_val_loss - self.best_val_loss  # positive = worse
            if loss_worsening > self.max_loss_spike_margin:
                self.should_stop = True
                self.print_to_log_file(
                    f"\n{'='*50}\n"
                    f"Stopping training: Dramatic loss spike detected!\n"
                    f"Best loss: {self.best_val_loss:.6f}\n"
                    f"Current loss: {current_val_loss:.6f}\n"
                    f"Worsening: +{loss_worsening:.4f} "
                    f"(threshold: +{self.max_loss_spike_margin})\n"
                    f"This suggests training instability.\n"
                    f"{'='*50}\n"
                )
                return

        # --- Improvement check -----------------------------------------------
        if current_val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
            self.nan_count = 0  # reset NaN counter on genuine improvement
            self.print_to_log_file(f"Validation loss improved to {current_val_loss:.6f}")
        else:
            self.epochs_without_improvement += 1
            self.print_to_log_file(
                f"No improvement for {self.epochs_without_improvement} epochs "
                f"(best: {self.best_val_loss:.6f}, current: {current_val_loss:.6f})"
            )

        # --- Early stopping check --------------------------------------------
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            self.print_to_log_file(
                f"\n{'='*50}\n"
                f"Early stopping triggered!\n"
                f"No improvement for {self.patience} epochs.\n"
                f"Best validation loss: {self.best_val_loss:.6f}\n"
                f"Stopping at epoch {self.current_epoch}\n"
                f"{'='*50}\n"
            )  # BUG FIX: removed duplicate identical print call that was here
    
    def run_training(self):
        """
        Override run_training to check for early stopping.
        When early stopping fires, the best checkpoint is restored before
        calling on_train_end() so that checkpoint_final.pth reflects the
        best weights rather than the last (possibly degraded) weights.
        """
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            if self.should_stop:
                self.print_to_log_file("Stopping training early...")
                # BUG FIX: restore best checkpoint so final weights = best weights,
                # not the last (potentially degraded) weights at stopping epoch.
                import os
                best_ckpt = os.path.join(self.output_folder, 'checkpoint_best.pth')
                if os.path.isfile(best_ckpt):
                    self.print_to_log_file(
                        f"Restoring best checkpoint from: {best_ckpt}"
                    )
                    self.load_checkpoint(best_ckpt)
                break

        self.on_train_end()


class nnUNetTrainerAdamEarlyStopping_patience25(nnUNetTrainerAdamEarlyStopping):
    """Adam with early stopping, patience=25"""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.patience = 25


class nnUNetTrainerAdamEarlyStopping_patience100(nnUNetTrainerAdamEarlyStopping):
    """Adam with early stopping, patience=100"""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.patience = 100


class nnUNetTrainerAdamEarlyStopping_patience200(nnUNetTrainerAdamEarlyStopping):
    """Adam with early stopping, patience=200"""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.patience = 200


class nnUNetTrainerAdamEarlyStopping_LowLR(nnUNetTrainerAdamEarlyStopping):
    """
    Adam with early stopping and lower learning rate for stability.
    Reduces NaN and loss spike issues.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4  # Lower than default 1e-2


class nnUNetTrainerAdamEarlyStopping_VeryLowLR(nnUNetTrainerAdamEarlyStopping):
    """
    Adam with early stopping and very low learning rate for maximum stability.
    Use if experiencing frequent NaN or dramatic loss spikes.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4  # Very conservative
        
        
class nnUNetTrainerAdamEarlyStopping_StrictGradClip(nnUNetTrainerAdamEarlyStopping):
    """
    Adam with early stopping and stricter gradient clipping.
    For cases with severe gradient explosion.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.max_grad_norm = 0.5  # Stricter clipping
        self.initial_lr = 3e-4  # Also lower LR for safety

