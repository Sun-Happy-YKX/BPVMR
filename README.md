
# 【Highlight🔥】BPVMR

The implementation of BPVMR.

In this paper, we creatively model ...

## ⚡ Demo

(暂未完成)

## 🌟 Overview

（暂未完成）

## 😍 Visualization

### Example 1

### Example 1

## 🚀 Quick Start

### Setup code environment

```sh
conda create -n HBI python=3.9
conda activate HBI
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

### Download Datasets

| Datasets     | Trained Data   | Test Data                                                             |
| ------------ | -------------- | --------------------------------------------------------------------- |
| Music-Dance  | Coming Soon... | [Download](https://pan.xunlei.com/s/VNZnwgUBLnxmw34YMYy1KZIeA1?pwd=ubuq) |
| Music-Motion | Proprietary    | [Download](https://pan.xunlei.com/s/VNZnwuKEp6fBzFJoNCEmeV2ZA1?pwd=tzve) |

This repository contains two types of the datasets of the project, that is, [Music-Dance]() and [Music-Motion](). Please note that at this time, we are only releasing the test sets for the [Music-Dance]() dataset, as the training sets are still being finalized. Once our paper has been accepted, we plan to release the complete training sets as well. Additionally, we are only able to showcase a portion of the videos from the [Music-Motion]() dataset, as this dataset was designed for another business of [Psycheai](https://www.psyai.com/home) and therefore cannot be fully disclosed.

### Directory Structure

After downloading the corresponding dataset, please move the relevant files to their respective directories. The 'output' folder represents the directory where the trained models are stored, the 'data' folder represents the directory where the [source data]() is located, and the 'checkout' folder represents the directory where the [similarity matrices]() for the training set are stored (used for QB-Norm operation).

The file directory  structure is as follows:

|—— checkout `<br>`
|       |── Data1 `<br>`
|       |── Data2 `<br>`
|—— config `<br>`
|—— data `<br>`
|       |── Music-Motion `<br>`
|       |── Music-Dance `<br>`
|—— datasets `<br>`
|—— models `<br>`
|—— modules `<br>`
|—— outputs `<br>`
|       |── Data1 `<br>`
|       |── Data2 `<br>`
|—— preprocess `<br>`
|—— trainer `<br>`
|—— order.sh `<br>`
|—— README.md `<br>`
|—— test_qb_norm.py `<br>`
|—— test.py `<br>`

### Result

You can directly get the reuslt of our paper by:

```
sh ./order.sh
```

## 🚀Training

```
Coming Soon...
```

## 🚀Evaluation

```shell
# QB-Norm mode for M-V Retrieval
python test_qb_norm.py --exp_name=Data2 --videos_dir=./data/Music-Dance --qbnorm_mode mode1 \
                --load_epoch -1 --use_beat --mode double_loss
# QB-Norm mode for V-M Retrieval
python test_qb_norm.py --exp_name=Data2 --videos_dir=./data/Music-Dance --qbnorm_mode mode1 \
                --load_epoch -1 --use_beat --mode double_loss --metric v2t
# Common mode for M-V Retrieval
python test.py --exp_name=Data2 --videos_dir=./data/Music-Dance \
                --load_epoch -1  --use_beat --mode double_loss
# Common mode for V-M Retrieval
python test.py --exp_name=Data2 --videos_dir=./data/Music-Dance \
                --load_epoch -1  --use_beat --mode double_loss --metric v2t
```

Currently, we are only releasing the [testing code](), and the [training code]() will be made available once the paper is accepted. In the testing code, we provide two testing scripts: '[test.py]()' for non-QB-Norm versions, and '[test_qb_norm.py]()' for QB-Norm versions. In 'test_qb_norm.py', you can adjust the 'mode' parameter to select the QB-Norm calculation mode.

## 🎗️ Acknowledgments

Our code is based on[ XPool](https://github.com/layer6ai-labs/xpool), and some of the data is provided by [Psycheai](https://www.psyai.com/home). We sincerely appreciate for their contributions.

## 📌 Citation

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```
@inproceedings{jin2023video,
  title={Video-text as game players: Hierarchical banzhaf interaction for cross-modal representation learning},
  author={Jin, Peng and Huang, Jinfa and Xiong, Pengfei and Tian, Shangxuan and Liu, Chang and Ji, Xiangyang and Yuan, Li and Chen, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2472--2482},
  year={2023}
}
```

(挂完arxiv之后再改一下)
