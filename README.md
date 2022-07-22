# Pix2Pix
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


This repository is an implementation of the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf).

I've implemented it reading the paper and comparing with the original authors/AladdinPersson implementation.

This Gan was trained to colorize anime pictures:

<p style="text-align: center;"> Inputs </p>

![input_81](https://user-images.githubusercontent.com/56324869/180009763-bf8a3e74-42a8-44ba-9bb3-3a8200135bf7.png)

<p style="text-align: center;"> Outputs </p>

![y_gen_81](https://user-images.githubusercontent.com/56324869/180009544-df49c1ed-7af7-4889-823b-efdf11978050.png)

The same model can be trained to generate maps from aerial images or vice-versa, and many other applications.

![image](https://user-images.githubusercontent.com/56324869/180003358-f12b9bed-be42-4abc-8ba2-022d7204a1f7.png)

---
## Training

I've trained it in the [anime-sketch-colorization-pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) dataset
- batch_size: 8 
- lr: 2e-4
- resolution: 256x256

BatchNorm2d changed for InstanceNorm2d (already changed by the authors in the CycleGAN paper):

Batchnorm computes one mean and std per batch, and make the whole Gaussian Unit. Instance norm computes one mean and std per sample in the batch, and then make each sample Gaussian Unit, separately. 
So using Instancenorm gives better visual results, specially on the background, since the background is pure white, where batchnorm may cause noise on it , since it makes the whole batch Gaussian Unit at once.

