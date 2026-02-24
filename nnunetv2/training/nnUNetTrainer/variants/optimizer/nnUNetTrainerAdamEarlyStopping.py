import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdam import nnUNetTrainerAdam


class nnUNetTrainerAdamEarlyStopping(nnUNetTrainerAdam):
    """
    Adam trainer with early stopping based on validation loss.
    
    Stops training if validation loss doesn't improve for `patience` epochs.
    Includes protection against NaN losses and gradient clipping for stability.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Initialize stability parameters BEFORE parent class
        self.patience = 50  # Number of epochs to wait for improvement
        self.min_delta = 1e-4  # Minimum change to qualify as improvement
        self.max_grad_norm = 1.0  # Gradient clipping max norm
        self.max_nan_tolerance = 3  # Stop after this many NaN occurrences
        self.max_loss_spike_ratio = 5.0  # Stop if loss spikes more than this ratio
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.nan_count = 0
        
        # Call parent without unpack_dataset (it's not in the signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
    
    def train_step(self, batch: dict) -> dict:
        """
        Override train step to add gradient clipping for stability.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            # Gradient clipping for stability
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}
        
    def on_validation_epoch_end(self, val_outputs: list):
        """
        Called after each validation epoch.
        Checks if we should stop training based on validation loss.
        Includes NaN detection and loss spike protection.
        """
        # Call parent method to compute metrics
        super().on_validation_epoch_end(val_outputs)
        
        # Get current validation loss
        current_val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
        
        # Check for NaN
        if np.isnan(current_val_loss) or np.isinf(current_val_loss):
            self.nan_count += 1
            self.print_to_log_file(
                f"\n{'!'*50}\n"
                f"WARNING: NaN or Inf loss detected! (occurrence {self.nan_count}/{self.max_nan_tolerance})\n"
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
        
        # Check for dramatic loss spike
        if self.best_val_loss != float('inf'):
            loss_ratio = current_val_loss / self.best_val_loss
            if loss_ratio > self.max_loss_spike_ratio:
                self.should_stop = True
                self.print_to_log_file(
                    f"\n{'='*50}\n"
                    f"Stopping training: Dramatic loss spike detected!\n"
                    f"Best loss: {self.best_val_loss:.6f}\n"
                    f"Current loss: {current_val_loss:.6f}\n"
                    f"Ratio: {loss_ratio:.2f}x (threshold: {self.max_loss_spike_ratio}x)\n"
                    f"This suggests training instability.\n"
                    f"{'='*50}\n"
                )
                return
        
        # Check if validation loss improved
        if current_val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
            self.nan_count = 0  # Reset NaN counter on improvement
            self.print_to_log_file(f"Validation loss improved to {current_val_loss:.6f}")
        else:
            self.epochs_without_improvement += 1
            self.print_to_log_file(
                f"No improvement for {self.epochs_without_improvement} epochs "
                f"(best: {self.best_val_loss:.6f}, current: {current_val_loss:.6f})"
            )
            
        # Check if we should stop
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            self.print_to_log_file(
                f"\n{'='*50}\n"
                f"Early stopping triggered!\n"
                f"No improvement for {self.patience} epochs.\n"
                f"Best validation loss: {self.best_val_loss:.6f}\n"
                f"Stopping at epoch {self.current_epoch}\n"
                f"{'='*50}\n"
            )
            self.print_to_log_file(
                f"\n{'='*50}\n"
                f"Early stopping triggered!\n"
                f"No improvement for {self.patience} epochs.\n"
                f"Best validation loss: {self.best_val_loss:.6f}\n"
                f"Stopping at epoch {self.current_epoch}\n"
                f"{'='*50}\n"
            )
    
    def run_training(self):
        """
        Override run_training to check for early stopping.
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
            
            # Check for early stopping
            if self.should_stop:
                self.print_to_log_file("Stopping training early...")
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

