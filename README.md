# Active Whitecap Extraction Using U-Net

This repository is for active whitecap extraction. It includes code for image stabilization and rectification based on horizon detection, model training and implement of pre-trained model. 

## Table of Contents

- [Dependencies](#1-dependencies)
- [Horizon Detection](#2-horizon-detection)
- [U-Net model training](#3-training)
- [Using a pre-trained model](#4-using-a-pre-trained-model)
- [Training dataset](#5-training-dataset)
- [Reference](#reference)

# 1 Dependencies

## Using conda:

- **Windows**

  ```bash
  conda env create -f dependencies.yml
  ```

## Manually:

```bash
# install git
conda install git
# create an environment
conda create -n whitecap_DL python=3.9
# activate the created environment
conda activate whitecap_DL
# if you want to use a nvidia GPU, please look at https://pytorch.org/get-started/locally/
# You must install cudnn and CUDA firstly and then install pytroch.
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# for CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y numpy tqdm pandas matplotlib 
pip install opencv-python
```
# 2 Horizon Detection


# 3 U-Net model training


# 4 Using a pre-trained model


# 5 Training dataset
https://drive.google.com/file/d/1MPFswZoO_TVPewWFIxUD5yxapu-4mMzj/view?usp=sharing
# Reference
https://github.com/cs230-stanford/cs230-code-examples
https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch
https://github.com/caiostringari/deepwaves