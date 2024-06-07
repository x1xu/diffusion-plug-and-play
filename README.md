# Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction
This repository is the official code base of the paper [Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction](https://arxiv.org/abs/2403.17042). 

Part of the code is adapted from [the repository for DPS](https://github.com/DPS2022/diffusion-posterior-sampling).

## Abstract
In a great number of tasks in science and engineering, the goal is to infer an unknown image from a small number of measurements collected from a known forward model describing certain sensing or imaging modality. Due to resource constraints, this task is often extremely ill-posed, which necessitates the adoption of expressive prior information to regularize the solution space. Score-based diffusion models, due to its impressive empirical success, have emerged as an appealing candidate of an expressive prior in image reconstruction. In order to accommodate diverse tasks at once, it is of great interest to develop efficient, consistent and robust algorithms that incorporate _unconditional_ score functions of an image prior distribution in conjunction with flexible choices of forward models.
This work develops an algorithmic framework for employing score-based diffusion models as an expressive data prior in general nonlinear inverse problems. Motivated by the plug-and-play framework in the imaging community, we introduce a diffusion plug-and-play method (DPnP) that alternatively calls two samplers, a proximal consistency sampler based solely on the likelihood function of the forward model, and a denoising diffusion sampler based solely on the score functions of the image prior. The key insight is that denoising under white Gaussian noise can be solved _rigorously_ via both stochastic (i.e., DDPM-type) and deterministic (i.e., DDIM-type) samplers using the unconditional score functions. We establish both asymptotic and non-asymptotic performance guarantees of DPnP, and provide numerical experiments to illustrate its promise in solving both linear and nonlinear image reconstruction tasks. To the best of our knowledge, DPnP is the first provably-robust posterior sampling method for nonlinear inverse problems using unconditional diffusion priors.

## Prerequisites
- python 3.10

- pytorch 2.2.0

- CUDA 12.1

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/x1xu/diffusion-pluy-and-play

cd diffusion-pluy-and-play
```

### 2) Download pretrained checkpoint
For FFHQ dataset, we use the score function from the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing). Download the checkpoint "ffhq_10m.pt" therein and paste it to `./models/`.
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

For ImageNet dataset, we use the score function from the [link](https://www.dropbox.com/scl/fi/0jwymqkdbppkfpuk2tmj6/imagenet256.pt). Again, download the checkpoint "imagenet256.pt" therein and paste it to `./models/`.


### 3) Set up environment
Install dependencies

```
conda create -n DPnP python=3.10

conda activate DPnP

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

```


### 4) Inference

```
python3 sample_condition.py \
--model_config=./configs/model_{MODEL-NAME}_config.yaml \
--diffusion_config=./configs/diffusion_config_{ALG-NAME}.yaml \
--task_config=./configs/{TASK-NAME}_{MODEL-NAME}_{ALG-NAME}.yaml;
```

where:
- `MODEL-NAME` can be one of `ffhq` and `imagenet`;
- `ALG-NAME` can be one of `dps`, `lgd`, and `pnp`;
- `TASK-NAME` can be one of `coded_pr` (for phase retrieval), `quantization` (for quantized sensing), and `super_resolution`.
<br />


### Structure of configurations
You need to write your data directory at data.root. Default is ./data/ffhq or ./data/imagenet which contains several sample images from FFHQ/ImageNet validation set.

```
conditioning:
    method: # ps, lgd, or mala
    params:
        # Algorithm-specific parameters; see corresponding .yaml file for examples 

data:
    name: general
    root: ./data/ffhq

measurement:
    operator:
        name: # coded_pr, quantization, or super_resolution

noise:
    name:   gaussian
    sigma:  # variance of Gaussian noise
```

## Citation
If you find our work interesting, please consider citing

```
@article{xu2024provably,
  title={Provably robust score-based diffusion posterior sampling for plug-and-play image reconstruction},
  author={Xu, Xingyu and Chi, Yuejie},
  journal={arXiv preprint arXiv:2403.17042},
  year={2024}
}
```

