# Parameterization-Based-NN-For-Nonlinear-Syntactic-Foam-Stress-Strain-Prediction

10/31/2022 By Haotian Feng and Pavana Prabhakar, University of Wisconsin - Madison

This repository includes the code and data for implementing the PBNN. [The paper can be find in this link](https://arxiv.org/abs/2212.12840). 

## Graphical Abstract
<img width="1149" alt="pbnn_new" src="https://user-images.githubusercontent.com/62448186/221620155-5cdc0d35-d8ad-4952-a75c-75f34b580f74.png">

## Implementation
In this repository, the PBNN framework is implemented by Tensorflow.
PBNN contains two modules: Feature Extraction Module (with idea of self-supervised learning) and Curve Prediction Module. For Curve Prediction Module, we implement two subframeworks: Cubic Function Representation and Ogden Function Representation. The only difference is that Ogden Function uses dynamic training process. 

## Folders
<ins>'Dataset'</ins> folder includes the dataset used for doing the analysis.

<ins>'Feature-Extraction-Size-Curve'</ins> folder includes the code for implementing Feature Extraction Module and testing the performance on different training sample sizes.

<ins>'PBNN_Implementation'</ins> folder includes the inplementation of PBNN when the curve is represented with Cubic Function or Ogden Function.


## Citation
To cite our paper, please use: 

`@article{feng2022parameterization,
  title={Parameterization-based Neural Network: Predicting Non-linear Stress-Strain Response of Composites},
  author={Feng, Haotian and Prabhakar, Pavana},
  journal={arXiv preprint arXiv:2212.12840},
  year={2022}
}`
