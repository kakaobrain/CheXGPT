import json
import time
from importlib import import_module
import pandas as pd
from torch import distributed as dist
from lightning.pytorch import LightningModule
from labeler.evaluation.chexgpt_metric import get_evaluator


class CxrLabelerLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self._sanity_check(cfg)
        self.cfg = cfg

        # Build a model
        module = import_module(f"labeler.model.{cfg.model.name}")
        self.model = module.Model(
            cfg.model.label_map,
            cfg.model.kwargs.p
        )

        # Prepare eval metrics
        #   - Update evaluator code, if cfg.head.label_map is changed
        self.evaluator = get_evaluator(cfg)

        # Extra variables
        self.all_predictions = {} # for saving prediction step outputs
        self._last_time = time.time() # for execution time logging

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        return self.model(input_ids, attention_mask)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._valtest_step(batch, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Run prediction on input batch. Store results in JSON Lines format.
        """
        output = self(batch)

        # Build output
        results = []
        for idx in range(len(batch["study_id"])):
            data = {}
            data["study_id"] = batch["study_id"][idx]
            for head_name in output:
                if output[head_name]["status"]["prediction_text"][idx] == "exist":
                    data[head_name] = {}
                    for attr_name in output[head_name]:
                        data[head_name][attr_name] = output[head_name][attr_name]["prediction_text"][idx]
            results.append(data)

        # Archieve partial result
        # TODO: Be careful not to run inference on too many samples! (~1M samples are okay)
        for r in results:
            self.all_predictions[str(r["study_id"])] = r

    def on_test_epoch_end(self):
        self._on_valtest_epoch_end("test")

    def on_predict_epoch_end(self):
        # Gather predictions across all GPUs
        output = [None for _ in range(self.trainer.world_size)]
        dist.gather_object(self.all_predictions, output if self.trainer.global_rank == 0 else None, dst=0)

        # Merge them and remove duplicates if exists
        all_preds = {}
        for preds in output:
            for k, v in preds.items():
                all_preds[k] = v

        # Print the result
        if self.global_rank == 0:
            output_path = self.cfg.predict.get("output_path", None)
            data_predict_cfg = list(self.cfg.data_predict.values())[0]
            data_format = data_predict_cfg.get("data_format", "jsonlines")

            if data_format == "jsonlines":
                if output_path is None:
                    output_path = "result.jsonl"
                with open(output_path, "w") as f:
                    for v in all_preds.values():
                        f.write(json.dumps(v) + "\n")

            elif data_format == "csv":
                if output_path is None:
                    output_path = "result.csv"
                out = []
                for row in all_preds.values():
                    study_id = row.pop("study_id")
                    status = []
                    for finding_name, finding_attrs in row.items():
                        if finding_attrs["status"] == "exist":
                            # Status
                            status.append(finding_name)
                    out.append([study_id, str(status)])
                pd.DataFrame(out, columns=["study_id", "status"]).to_csv(output_path, index=False)

            else:
                raise ValueError(f"Unknown data_format: {data_format}")

    def _sanity_check(self, cfg):
        for k in cfg.head.label_map:
            # Each head (finding) must have "status" attribute
            assert "status" in cfg.head.label_map[k], "status must be specified"
            assert len(cfg.head.label_map[k]["status"]["values"]) == 2, "status must be a list of two elements"
            assert "exist" in cfg.head.label_map[k]["status"]["values"], "status must contain 'exist'"
            assert "not_exist" in cfg.head.label_map[k]["status"]["values"], "status must contain 'not_exist'"

    def _valtest_step(self, batch, dataloader_idx=0):
        output = self(batch)

        # Update FC evaluator(s)
        self.evaluator.update(output, batch["labels"], dataloader_idx)

    def _on_valtest_epoch_end(self, run_type):
        results = self.evaluator.get_results(run_type)

        # Log on TB
        log_options = {"on_step": False, "on_epoch": True, "sync_dist": True}
        for k, v in results.items():
            self.log(k, v, **log_options) # TB logging

        # Log on console
        self._print_results_on_console(results)

    def _print_results_on_console(self, results):
        """Print results on conosole screen
        """
        # Format data
        output = {}
        for ks, v in results.items():
            # ks = "{data_type}/{dataset_idx}/{metric_type}/{category}/{score_type}"
            ks = ks.split("/")
            _tmp = output
            for k in ks[:-1]:
                if k not in _tmp:
                    _tmp[k] = {}
                _tmp = _tmp[k]
            _tmp[ks[-1]] = v

        # Print data
        for data_type in output:
            for dataset_idx in output[data_type]:
                for metric_type in output[data_type][dataset_idx]:
                    met = output[data_type][dataset_idx][metric_type]
                    if metric_type == "status":
                        msg = self._build_status_output(data_type, dataset_idx, metric_type, met)
                    else:
                        raise ValueError(f"Unknown metric_type: {metric_type}")
                    self.print(msg)

    def _build_status_output(self, data_type, dataset_idx, metric_type, result):

        categories = sorted(result.keys())

        msg = f"# {data_type}/{dataset_idx}/{metric_type}\n\n"
        msg += "Category | F1 | Precision | Recall |\n"
        msg += "| -- | -- | -- | -- |\n"

        for category in categories:
            msg += f"{category}  | "
            for score_type in ["f1", "prec", "rec"]:
                msg += f"{result[category][score_type] * 100:.2f} | "
            msg += "\n"

        msg += "\n"

        return msg

    def _log_on_console(self, loss):

        is_logging_step = self.global_step % self.trainer.log_every_n_steps == 0
        if is_logging_step:
            # Calculate the elapsed time
            elapsed_time = time.time() - self._last_time
            self._last_time = time.time()
            self.print(f"[{self.global_step}/{self.trainer.max_steps}] ({elapsed_time:.0f}s) loss: {loss.item():.3f}")
