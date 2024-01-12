import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Model(nn.Module):
    def __init__(self, label_map, p=0.1):
        super().__init__()

        self.label_map = label_map
        self.p = p

        # Prepare a backbone
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Disable unnecessary "BertPooler" layer
        hidden_size = self.bert.pooler.dense.in_features
        self.bert.pooler = None

        # Prepare heads
        self.dropout = nn.Dropout(p)

        self.linear_heads = nn.ModuleDict()
        for head_name, attrs in label_map.items():
            attr_heads = {}
            for attr_name, attr_info in attrs.items():
                attr_heads[attr_name] = nn.Linear(hidden_size, len(attr_info["values"]), bias=True)
            self.linear_heads[head_name] = nn.ModuleDict(attr_heads)

    def forward(self, source_padded, attention_mask):
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]

        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)

        out = {head_name: {} for head_name in self.label_map}
        for head_name, attrs in self.label_map.items():
            for attr_name, attr_info in attrs.items():
                item = {}

                item["logits"] = self.linear_heads[head_name][attr_name](cls_hidden)
                item["prediction_id"] = [_id for _id in item["logits"].argmax(-1).tolist()]
                item["prediction_text"] = [attr_info["values"][_id] for _id in item["prediction_id"]]

                out[head_name][attr_name] = item

        return out

    def get_tokenizer(self):
        return self.tokenizer
