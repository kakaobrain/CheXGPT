import torch
from torch import nn
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall


def get_evaluator(cfg):
    if cfg.mode == "val":
        datasets = cfg.data_val
    elif cfg.mode == "test":
        datasets = cfg.data_test
    else:
        datasets = []

    return OfflineGptMetric(cfg.head, len(datasets))


def safe_division(x, y):
    if y > 0.0:
        return x / y
    else:
        return 0.0


class OfflineGptMetric(nn.Module):
    def __init__(self, head_cfg, num_datasets):
        super().__init__()

        self.metrics = nn.ModuleList()
        self.num_datasets = num_datasets

        self.label_map = head_cfg.label_map
        self.status_name_list = list(self.label_map.keys())
        self.num_status = len(self.status_name_list)

        for idx in range(num_datasets):
            self.metrics.append(nn.ModuleDict())

            self.metrics[idx]["status"] = nn.ModuleDict({
                "prec": MultilabelPrecision(num_labels=self.num_status, average=None),
                "rec": MultilabelRecall(num_labels=self.num_status, average=None),
                "f1": MultilabelF1Score(num_labels=self.num_status, average=None),
            })

    @torch.inference_mode()
    def update(self, output, label, dataloader_idx):
        metric = self.metrics[dataloader_idx]["status"]
        new_output, new_label = self._build_status_only_output_and_label(output, label)
        self._update_metric(metric, new_output, new_label)

    def _build_status_only_output_and_label(self, output, label):
        new_output = []
        new_label = []

        for head_name, attrs in self.label_map.items():
            new_output.append(output[head_name]["status"]["logits"].argmax(-1))
            new_label.append(label[head_name]["status"].argmax(-1))

        new_output = torch.stack(new_output).T
        new_label = torch.stack(new_label).T

        return new_output, new_label

    def _update_metric(self, metric, output, label):
        if output.shape[0] != 0:
            metric["f1"](output, label)
            metric["prec"](output, label)
            metric["rec"](output, label)

    @torch.inference_mode()
    def get_results(self, run_type):
        results = {}
        for ds_idx, ds_metrics in enumerate(self.metrics):
            res = self._compute_status_results(
                run_type,
                ds_idx,
                ds_metrics["status"],
                "status",
                self.status_name_list)
            results = {**results, **res}

        return results

    def _compute_status_results(self, run_type, ds_idx, met, met_type, status_names):
        target_classes = [
            'widened mediastinal silhouette', 'nodule', 'lung opacity', 'pulmonary edema',
            'consolidation', 'atelectasis', 'pneumothorax', 'effusion', 'pleural lesion', 'fracture'
        ]

        results = {}

        prec_scores = met["prec"].compute().cpu().numpy()
        rec_scores = met["rec"].compute().cpu().numpy()
        f1_scores = met["f1"].compute().cpu().numpy()

        # Macro-average
        prefix = f"{run_type}/{ds_idx}/{met_type}/macro"
        results[prefix + "/prec"] = prec_scores.mean()
        results[prefix + "/rec"] = rec_scores.mean()
        results[prefix + "/f1"] = f1_scores.mean()

        if len(status_names) == 13:
            prefix = f"{run_type}/{ds_idx}/{met_type}/macro_10_classes"
            results[prefix + "/prec"] = 0.0
            results[prefix + "/rec"] = 0.0
            results[prefix + "/f1"] = 0.0
            for h_idx, status_name in enumerate(status_names):
                if status_name in target_classes:
                    results[prefix + "/prec"] += prec_scores[h_idx]
                    results[prefix + "/rec"] += rec_scores[h_idx]
                    results[prefix + "/f1"] += f1_scores[h_idx]
            results[prefix + "/prec"] /= 10
            results[prefix + "/rec"] /= 10
            results[prefix + "/f1"] /= 10

        # Micro-average
        prefix = f"{run_type}/{ds_idx}/{met_type}/micro"
        tp, fp, fn = 0, 0, 0
        for h_idx, status_name in enumerate(status_names):
            tp += met["f1"].tp[h_idx].item()
            fp += met["f1"].fp[h_idx].item()
            fn += met["f1"].fn[h_idx].item()
        _prec = safe_division(tp, tp + fp)
        _rec = safe_division(tp, tp + fn)
        results[prefix + "/prec"] = _prec
        results[prefix + "/rec"] = _rec
        results[prefix + "/f1"] = 2 * _prec * _rec / (_prec + _rec)

        if len(status_names) == 13:
            prefix = f"{run_type}/{ds_idx}/{met_type}/micro_10_classes"
            tp, fp, fn = 0, 0, 0
            for h_idx, status_name in enumerate(status_names):
                if status_name in target_classes:
                    tp += met["f1"].tp[h_idx].item()
                    fp += met["f1"].fp[h_idx].item()
                    fn += met["f1"].fn[h_idx].item()
            _prec = safe_division(tp, tp + fp)
            _rec = safe_division(tp, tp + fn)
            results[prefix + "/prec"] = _prec
            results[prefix + "/rec"] = _rec
            results[prefix + "/f1"] = 2 * _prec * _rec / (_prec + _rec)

        # per class
        for h_idx, status_name in enumerate(status_names):
            # Precision, Recall, and F1-score
            prefix = f"{run_type}/{ds_idx}/{met_type}/{status_name}"
            results[prefix + "/prec"] = prec_scores[h_idx]
            results[prefix + "/rec"] = rec_scores[h_idx]
            results[prefix + "/f1"] = f1_scores[h_idx]

            # FP ratio (additional information for analysis)
            f1_metric = met["f1"]
            fp_ratio = f1_metric.fp[h_idx].float() / (f1_metric.fp[h_idx] + f1_metric.tn[h_idx])
            results[prefix + "/fp"] = fp_ratio.item()

        met["prec"].reset()
        met["rec"].reset()
        met["f1"].reset()

        return results
