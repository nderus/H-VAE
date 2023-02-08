# Synthetic Data Generation of histopathological images
This repository contains the implementation for the work "Synthetic Data Generation Of Histopathological Images" presented at the XXII International Conference on Mechanics in Medicine and Biology, in Bologna (2022).<br />
The data set used in the study is publicly available for download at: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images). In order to launch the training scripts, it is assumed that the data is available in the directory: 'datasets/breast-histopathology/IDC_regular_ps50_idx5'.<br />
The data set in tfds format is available here: [Google Drive](https://drive.google.com/drive/folders/1UHCs9GCNUZzjSxcdjXc3gp-QpniFfVGd?usp=sharing)

---
## Overview
## Training
Please refer to the scripts provided in the table corresponding to some training tasks possible using the code.

|          **Task**          	|      **Reference**      	|
|:--------------------------:	|:-----------------------:	|
| Training VAE (1st stage)  	|  `training/vae_train.py`  |
| Training DDPM (2nd stage) 	|  `training/ddpm_train.py`	|
| Classifier (Diagnostic)     |  `training/classifier.py` |

## Training
![training-2](https://user-images.githubusercontent.com/99331278/191777709-52d8a58b-bd35-449e-9ecf-298589e366a1.png)

## Generation

![generation-2](https://user-images.githubusercontent.com/99331278/191778151-e4c97754-56a7-46f4-b908-44ca882a63ae.png)

---

`conda create --name H-VAE --file requirements.txt` 

---

Since our model uses diffusion models please consider citing the original [DiffuseVAE](https://arxiv.org/abs/2201.00308) [Diffusion model](https://arxiv.org/abs/1503.03585), [DDPM](https://arxiv.org/abs/2006.11239) and [VAE](https://arxiv.org/abs/1312.6114) papers.
