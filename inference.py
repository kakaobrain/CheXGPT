import torch
import yaml
from collections import OrderedDict
from labeler.model.chexgpt_labeler import Model

HEAD_CONFIG_PATH = "./configs/head/evaluation_chexgpt.yaml"
PRETRAIN_PATH = "./checkpoint/model_mixed.ckpt"

# read head configuration
with open(HEAD_CONFIG_PATH) as f:
    head_cfg = yaml.load(f, Loader=yaml.FullLoader)
label_map = head_cfg["label_map"]

# load model
model = Model(label_map).eval()
ckpt = torch.load(PRETRAIN_PATH, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    name = k[6:]  # remove `model.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=True)
tokenizer = model.get_tokenizer()

# inference
inputs = ["Focal consolidation is identified.",
          "There is likely some atelectasis at the right base.",
          "Displaced left seventh posterior rib fracture.",
          "The lungs are hyperinflated compatible with underlying COPD.",
          "The heart is enlarged.",
          ]
tokenized = tokenizer(
    inputs,
    padding=True,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
    pad_to_multiple_of=8)

with torch.inference_mode():
    outputs = model(tokenized.input_ids, tokenized.attention_mask)

# display outputs
for head_name, attrs in label_map.items():
    for attr_name, attr_info in attrs.items():
        text = outputs[head_name][attr_name]["prediction_text"]
        print(f"{head_name} | {attr_name} | {text}")
# atelectasis | status | ['not_exist', 'exist', 'not_exist', 'not_exist', 'not_exist']
# consolidation | status | ['exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# effusion | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# fracture | status | ['not_exist', 'not_exist', 'exist', 'not_exist', 'not_exist']
# hyperinflation | status | ['not_exist', 'not_exist', 'not_exist', 'exist', 'not_exist']
# lung opacity | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# nodule | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# pleural lesion | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# pneumothorax | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# pulmonary edema | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# subcutaneous emphysema | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# subdiaphragmatic gas | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'not_exist']
# widened mediastinal silhouette | status | ['not_exist', 'not_exist', 'not_exist', 'not_exist', 'exist']
