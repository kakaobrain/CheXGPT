import json
from copy import deepcopy
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class CxrLabelerDataModule(LightningDataModule):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

        # Prepare a collate function
        self.collate_fn = CxrLabelerDataCollator(tokenizer)

    def setup(self, stage):
        if stage == "test":
            self.test = [CxrLabelerDataset(self.cfg.data_test[name]) for name in self.cfg.data_test]

        if stage == "predict":
            assert len(self.cfg.data_predict) == 1, "predict mode supports only one data!"
            self.predict = [CxrLabelerDataset(self.cfg.data_predict[name]) for name in self.cfg.data_predict]

    def test_dataloader(self):
        dataloaders = [DataLoader(ds, collate_fn=self.collate_fn, **self.cfg.dataloader.test) for ds in self.test]
        return dataloaders

    def predict_dataloader(self):
        dataloaders = [DataLoader(ds, collate_fn=self.collate_fn, **self.cfg.dataloader.predict) for ds in self.predict]
        return dataloaders


class CxrLabelerDataset(Dataset):
    def __init__(self, data_config):
        self.data_config = data_config

        # Prepare label2index and index2label mapping
        self.label2index = {k: {} for k in data_config.label_map}
        for head_name, attrs in data_config.label_map.items():
            for attr_name, attr_info in attrs.items():
                self.label2index[head_name][attr_name] = {v: i for i, v in enumerate(attr_info["values"])}

        self.index2label = deepcopy(self.label2index)
        for head_name, attrs in self.index2label.items():
            for attr_name, label2index in attrs.items():
                self.index2label[head_name][attr_name] = {v: k for k, v in label2index.items()}

        # Load data
        data_format = data_config.get("data_format", "jsonlines")
        if data_format == "jsonlines":
            with open(data_config.data_path) as f:
                self.data = pd.DataFrame([json.loads(line) for line in f])
        elif data_format == "csv":
            self.data = pd.read_csv(data_config.data_path, index_col=None).fillna("")
        else:
            raise ValueError(f"Unknown data format: {data_format}")

    def __getitem__(self, index):
        sample = self.data.iloc[index]

        # 1. Get study_id
        study_id = int(sample["study_id"]) if str(sample["study_id"]).isdigit() else sample["study_id"]

        # 2. Get text
        if "text" in sample and len(sample["text"]) > 0:
            text = sample["text"]
        else:
            raise ValueError("No text column found in the sample")

        # 3. Get labels
        if self.data_config.label_type == "none":
            indexed_label = {}
            raw_label = {}
        else:
            label_map = self.data_config.label_map
            input_label = sample[self.data_config.label_type]
            indexed_label = {k: {} for k in self.data_config.label_map} # For model
            raw_label = {k: {} for k in self.data_config.label_map} # For logging

            for head_name, attrs in label_map.items():
                for attr_name, attr_value in attrs.items():
                    # Get attribute label(s)
                    label_values = input_label.get(head_name, {}).get(attr_name, attr_value["default"])

                    # Get a target tensor of the attribute labels
                    attr_label_tensor = [0.0] * len(self.label2index[head_name][attr_name])
                    for attr_value in label_values:
                        idx = self.label2index[head_name][attr_name][attr_value]
                        attr_label_tensor[idx] = 1.0

                    # Save the target tensor
                    indexed_label[head_name][attr_name] = attr_label_tensor
                    raw_label[head_name][attr_name] = label_values

        data = {
            "study_id": study_id,
            "text": text,
            "label": indexed_label,
            "raw_label": raw_label,
        }

        return data

    def __len__(self):
        return len(self.data)


class CxrLabelerDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        raw_inputs = [sample["text"] for sample in samples]
        raw_labels = [sample["raw_label"] for sample in samples]
        indexed_labels = [sample["label"] for sample in samples]
        study_ids = [sample["study_id"] for sample in samples]

        # Tokenize input texts
        tokenized_inputs = self.tokenizer(
            raw_inputs,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            pad_to_multiple_of=8)

        # Collate labels for each (head_name, attr_name)
        batch_indexed_labels = {k: {} for k in indexed_labels[0].keys()}
        for indexed_label in indexed_labels:
            for head_name, attrs in indexed_label.items():
                for attr_name, attr_label in attrs.items():
                    if attr_name not in batch_indexed_labels[head_name]:
                        batch_indexed_labels[head_name][attr_name] = [attr_label]
                    else:
                        batch_indexed_labels[head_name][attr_name].append(attr_label)
        for head_name, attrs in batch_indexed_labels.items():
            for attr_name, attr_label in attrs.items():
                batch_indexed_labels[head_name][attr_name] = torch.tensor(attr_label)

        data = {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask,
            "labels": batch_indexed_labels,
            "study_id": study_ids,
            "raw_inputs": raw_inputs,
            "raw_labels": raw_labels,
        }

        return data
