# Synthetic Data Generation of histopathological images
This repository contains the implementation for the work "SYNTHETIC DATA GENERATION OF HISTOPATHOLOGICAL IMAGES" presented at the XXII International Conference on Mechanics in Medicine and Biology, in Bologna (2022).
The data set used in the study is publicly available for download at: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images. In order to launch the training scripts, it is assumed that the data is available in the directory: 'datasets/breast-histopathology/IDC_regular_ps50_idx5'.

The data set in tfds format is available here: https://drive.google.com/drive/folders/1UHCs9GCNUZzjSxcdjXc3gp-QpniFfVGd?usp=sharing
---
## Overview
## Training
![training-2](https://user-images.githubusercontent.com/99331278/191777709-52d8a58b-bd35-449e-9ecf-298589e366a1.png)

## Generation

![generation-2](https://user-images.githubusercontent.com/99331278/191778151-e4c97754-56a7-46f4-b908-44ca882a63ae.png)

---

Since our model uses diffusion models please consider citing the original [DiffuseVAE](https://arxiv.org/abs/2201.00308) [Diffusion model](https://arxiv.org/abs/1503.03585), [DDPM](https://arxiv.org/abs/2006.11239) and [VAE](https://arxiv.org/abs/1312.6114) papers.
