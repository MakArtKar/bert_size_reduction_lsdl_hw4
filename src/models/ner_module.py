import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from seqeval.metrics import f1_score
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import AutoModelForTokenClassification

from src.metrics.f1_score_seqeval import SeqevalF1Score
from src.models.components.factorized_embedding_wrapper import modify_bert_with_factorized_embedding

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NERLitModule(LightningModule):
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model_name: str = 'bert-base-cased',
        factorized_embeddings_hidden_size: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(self.label_names))

        # metric objects for calculating and averaging accuracy across batches
        self.metrics = torch.nn.ModuleDict({
            'fit': SeqevalF1Score(self.label_names),
            'val': SeqevalF1Score(self.label_names),
            'test': SeqevalF1Score(self.label_names),
        })

        # for averaging loss across batches
        self.losses = torch.nn.ModuleDict({
            'fit': MeanMetric(),
            'val': MeanMetric(),
            'test': MeanMetric(),
        })

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.net(**x)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.losses['val'].reset()
        self.metrics['val'].reset()
        self.val_acc_best.reset()

        if self.hparams.factorized_embeddings_hidden_size is not None:
            self.net = modify_bert_with_factorized_embedding(self.net, self.hparams.factorized_embeddings_hidden_size)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(batch)
        logits, loss = output.logits, output.loss
        preds = torch.argmax(logits, dim=-1)

        self.losses[mode](loss)
        self.metrics[mode](preds, batch['labels'])
        self.log(f"{mode}/loss", self.losses[mode], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/f1_score", self.metrics[mode], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.model_step(batch, 'fit')

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        return self.model_step(batch, 'val')

    def on_validation_epoch_end(self) -> None:
        acc = self.metrics['val'].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_score_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

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
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
