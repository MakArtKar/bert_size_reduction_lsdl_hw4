import torch
from torchmetrics import Metric
from seqeval.metrics import f1_score

class SeqevalF1Score(Metric):
    def __init__(self, label_names, **kwargs):
        super().__init__(compute_on_step=False)
        self.label_names = label_names
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.kwargs = kwargs

    def update(self, preds, targets):
        """
        Updates the state with predictions and targets.
        Args:
            preds: List of predicted labels for each sequence.
            targets: List of ground truth labels for each sequence.
        """
        # Detach tensors and convert them to Python lists
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().tolist()
            preds = [self.label_names[idx] for idx in preds]
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().tolist()
            targets = [self.label_names[idx] for idx in targets]
        
        # Append predictions and targets
        self.predictions.extend(preds)
        self.targets.extend(targets)

    def compute(self):
        """
        Computes the F1 score using seqeval.
        """
        return f1_score(self.targets, self.predictions, **self.kwargs)
