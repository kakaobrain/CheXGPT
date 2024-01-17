[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)

# CheX-GPT
This is an official inference code of **"CheX-GPT: Harnessing Large Language Models for Enhanced Chest X-ray Report Labeling"** [[arxiv]](https://arxiv.org/)

## Environment setup
We have experimented the implementation on the following enviornment.
- Python 3.11
- CUDA 12.1
```bash
conda create -n chexgpt python=3.11
conda activate chexgpt
pip install -r requirements.txt
```

## Prepare dataset
TBU - a subset of MIMIC test data (500 reports and paired labels)

## Model checkpoint
Download the model checkpoint from here [[link]](https://twg.kakaocdn.net/brainrepo/models/CheX-GPT/model_mixed.ckpt) and place the model in the 'checkpoint' directory.


## Command line
* Test 
  * CE metrics are displayed
      ```bash
      python main.py mode=test
      ```
* Predict 
  * Labeler outputs are saved in jsonline format
      ```bash
      python main.py mode=predict predict.output_path=${save_path}
      ```
* Inference
  * You can directly input CXR reports and check the labeler output.
    ```bash
    python inference.py
    ```
  
## Citation
TBU

## License
- The source code of CheX-GPT is licensed under [CC-BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
- The CheX-GPT model initialization utilized the [ChexBert](https://github.com/stanfordmlgroup/CheXbert). For information regarding the licensing of ChexBert, please refer to the following links:
  - [ChexBert License](https://github.com/stanfordmlgroup/CheXbert/blob/master/LICENSE.pdf)
  - [Stanford Office of Technology Licensing](http://techfinder2.stanford.edu/technology_detail.php?ID=43869)

## Contact
- Jawook Gu, [jawook.gu@kakaobrain.com](jawook.gu@kakaobrain.com)
- Kihyun You, [ukihyun@kakaobrain.com](ukihyun@kakaobrain.com)  