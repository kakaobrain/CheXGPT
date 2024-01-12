[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)

# CheX-GPT
This is an official inference code of **"CheX-GPT: Harnessing Large Language Models for Enhanced Chest X-ray Report Labeling"** [[arxiv]](https://arxiv.org/)

## Environment setup
We have experimented the implementation on the following enviornment.
- Python 3.11
- CUDA 12.1
```bash
pip install -r requirements.txt
```

## Prepare dataset
TBU - a subset of MIMIC test data (500 reports and paired labels)

## Model checkpoint
TBU - CheX-GPT model download link

## Command line
* Test code (to check CE metrics)
    ```bash
    python main.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_FILENAME} mode=test
    ```
* Predict (to get labeler outputs)
    ```bash
    python main.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_FILENAME} mode=predict
    ```

## Citation
TBU

## License
TBU

## Contact
- Jawook Gu, [jawook.gu@kakaobrain.com](jawook.gu@kakaobrain.com)
- Kihyun You, [ukihyun@kakaobrain.com](ukihyun@kakaobrain.com)  