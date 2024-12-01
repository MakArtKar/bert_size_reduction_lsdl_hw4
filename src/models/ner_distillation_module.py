from typing import Any, Dict, Tuple, Optional, Union, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from transformers import AutoModelForTokenClassification

from src.models.ner_module import NERLitModule


class NERDistillationLitModule(NERLitModule):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model_name: Union[str, nn.Module] = 'prajjwal1/bert-small',
        teacher_model_name: Union[str, nn.Module] = 'bert-base-cased',
        model_processors: List[Callable] = [],
        temperature: float = 5,
    ) -> None:
        super().__init__(optimizer, scheduler, compile, model_name, model_processors)

        if isinstance(teacher_model_name, str):
            self.teacher = AutoModelForTokenClassification.from_pretrained(teacher_model_name, num_labels=len(self.label_names))
        else:
            self.teacher = teacher_model_name

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.temperature = temperature

        self.distil_criterion = torch.nn.KLDivLoss(reduction="sum", log_target=True)

        self.losses = torch.nn.ModuleDict({
            'fit_distil': MeanMetric(),
            'val_distil': MeanMetric(),
            'test_distil': MeanMetric(),
            **self.losses._modules,
        })

    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(**x)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.losses['val_distil'].reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, logits, preds = super().model_step(batch, mode)
        teacher_logits = self.forward_teacher(batch).logits
        
        soft_student = F.log_softmax(logits / self.temperature, dim=-1)
        soft_teacher = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        mask = ((batch['attention_mask'].roll(1)) & (batch['attention_mask'].roll(-1))).bool()
        distil_loss = self.distil_criterion(
            soft_student[mask], soft_teacher[mask]
        ) * (self.temperature ** 2) / teacher_logits.size(0)

        self.losses[f'{mode}_distil'](distil_loss)
        self.log(f"{mode}/distil_loss", self.losses[f'{mode}_distil'], on_step=False, on_epoch=True, prog_bar=True)
        alpha = self.current_epoch / self.trainer.max_epochs
        if mode == 'train':
            final_loss = (1 - alpha) * distil_loss + alpha * loss
        else:
            final_loss = distil_loss + loss
        return final_loss, logits, preds

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.teacher = torch.compile(self.teacher)
