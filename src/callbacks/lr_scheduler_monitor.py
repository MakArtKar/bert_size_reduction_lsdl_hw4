from lightning.pytorch.callbacks import Callback

class LRSchedulerMonitor(Callback):
    def __init__(self, log_name: str = "lr", interval: str = "epoch"):
        """
        Monitor and log learning rate from a scheduler during training.
        
        Args:
            log_name (str): Name of the logged learning rate. Default is "lr".
            interval (str): Logging interval, either "epoch" or "step". Default is "epoch".
        """
        super().__init__()
        self.log_name = log_name
        self.interval = interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.interval == "step":
            self._log_lr(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.interval == "epoch":
            self._log_lr(trainer, pl_module)

    def _log_lr(self, trainer, pl_module):
        # Get optimizer and scheduler
        for i, optimizer in enumerate(trainer.optimizers):
            if hasattr(optimizer, "param_groups"):
                for j, group in enumerate(optimizer.param_groups):
                    # Extract learning rate and log it
                    lr = group.get("lr", None)
                    if lr is not None:
                        log_key = f"{self.log_name}/optimizer_{i}_group_{j}"
                        trainer.logger.log_metrics({log_key: lr}, step=trainer.global_step)
