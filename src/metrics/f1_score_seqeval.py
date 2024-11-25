import torch
from torchmetrics import Metric
from seqeval.metrics import f1_score

class SeqevalF1Score(Metric):
    def __init__(self, label_names, ignore_index=-100, **kwargs):
        super().__init__()
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

        self.label_names = label_names
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def update(self, preds, targets):
        """
        Updates the state with predictions and targets.
        Args:
            preds: List of predicted labels for each sequence.
            targets: List of ground truth labels for each sequence.
        """
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        
        for pred, target in zip(preds, targets):
            pred_labels = torch.tensor([p for p, t in zip(pred, target) if t != self.ignore_index])
            target_labels = torch.tensor([t for t in target if t != self.ignore_index])

            self.predictions.append(pred_labels)
            self.targets.append(target_labels)

    def compute(self):
        """
        Computes the F1 score using seqeval.
        """
        targets = [[self.label_names[t] for t in target] for target in self.targets]
        predictions = [[self.label_names[t] for t in prediction] for prediction in self.predictions]
        return torch.tensor([f1_score(targets, predictions, **self.kwargs)])
