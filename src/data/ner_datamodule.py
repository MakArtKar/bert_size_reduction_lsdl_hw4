from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def custom_collate_fn(batch):
    input_ids = pad_sequence([torch.LongTensor(item["input_ids"]) for item in batch], batch_first=True)
    attention_mask = pad_sequence([torch.LongTensor(item["attention_mask"]) for item in batch], batch_first=True)
    ner_tags = pad_sequence([torch.LongTensor(item["ner_tags"]) for item in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": ner_tags}


class NERDataModule(LightningDataModule):

    def __init__(
        self,
        batch_size: int = 64,
        dataloader_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def align_labels_with_tokens(self, example: Dict[str, Any]) -> Dict[str, Any]:
        tokenized_text = self.tokenizer(example["tokens"], is_split_into_words=True)
        aligned_labels = []
        prev_word_id = None
        for word_id in tokenized_text.word_ids():
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(example["ner_tags"][word_id])
            else:
                aligned_labels.append((example["ner_tags"][word_id] + 1) // 2 * 2)
            prev_word_id = word_id
        tokenized_text["ner_tags"] = aligned_labels
        return tokenized_text

    def prepare_data(self) -> None:
        dataset = load_dataset("eriktks/conll2003")

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = load_dataset("eriktks/conll2003")
            dataset = dataset.map(self.align_labels_with_tokens).remove_columns(["id", "pos_tags", "chunk_tags", "tokens"])

            self.data_train = dataset['train']
            self.data_val = dataset['validation']
            self.data_test = dataset['test']

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            collate_fn=custom_collate_fn,
            **self.hparams.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=custom_collate_fn,
            **self.hparams.dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=custom_collate_fn,
            **self.hparams.dataloader_kwargs,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()
