from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from seqeval.metrics import f1_score
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import AutoModelForTokenClassification

from src.metrics.f1_score_seqeval import SeqevalF1Score


class NERLitModule(LightningModule):
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model_name: str = 'bert-base-cased',
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(self.label_names))

        # loss function
        self.critertion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.accs = torch.nn.ModuleDict({
            'train': SeqevalF1Score(self.label_names),
            'val': SeqevalF1Score(self.label_names),
            'test': SeqevalF1Score(self.label_names),
        })

        # for averaging loss across batches
        self.losses = torch.nn.ModuleDict({
            'train': MeanMetric(),
            'val': MeanMetric(),
            'test': MeanMetric(),
        })

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.losses['val'].reset()
        self.accs['val'].reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch['input_ids'], batch['ner_tags']
        logits = self.forward(x)
        loss = self.criterion(logits_student, y)
        preds = torch.argmax(logits_student, dim=1)

        self.losses[mode](loss)
        self.accs[mode](preds, y)
        self.log(f"{mode}/loss", self.losses[mode], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/acc", self.accs[mode], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.model_step(batch, 'train')

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        return self.model_step(batch, 'val')

    def on_validation_epoch_end(self) -> None:
        acc = self.accs['val'].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        return self.model_step(batch, 'test')

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
