# Synthetic Data Generation of histopathological images
This repository contains the implementation for the work "SYNTHETIC DATA GENERATION OF HISTOPATHOLOGICAL IMAGES" presented at the XXII International Conference on Mechanics in Medicine and Biology, in Bologna (2022).

---
## Overview
## Training
![training-2](https://user-images.githubusercontent.com/99331278/191777709-52d8a58b-bd35-449e-9ecf-298589e366a1.png)

Please refer to the scripts provided in the table corresponding to some training tasks possible using the code.

|          **Task**          	|      **Reference**      	|
|:--------------------------:	|:-----------------------:	|
|  Training First stage VAE  	|  `scripts/train_ae.sh`  	|
| Training Second stage DDPM 	| `scripts/train_ddpm.sh` 	|


## Generation

![generation-2](https://user-images.githubusercontent.com/99331278/191778151-e4c97754-56a7-46f4-b908-44ca882a63ae.png)

Please refer to the scripts provided in the table corresponding to some inference tasks possible using the code.

|                          **Task**                         	|         **Reference**         	|
|:---------------------------------------------------------:	|:-----------------------------:	|
|            Sample/Reconstruct from Baseline VAE           	|      `scripts/test_ae.sh`     	|
|                   Sample from DiffuseVAE                  	|     `scripts/test_ddpm.sh`    	|
|          Generate reconstructions from DiffuseVAE         	| `scripts/test_recons_ddpm.sh` 	|
| Interpolate in the VAE/DDPM latent space using DiffuseVAE 	|    `scripts/interpolate.sh`   	|

---

Since our model uses diffusion models please consider citing the original [DiffuseVAE](https://arxiv.org/abs/2201.00308) [Diffusion model](https://arxiv.org/abs/1503.03585), [DDPM](https://arxiv.org/abs/2006.11239) and [VAE](https://arxiv.org/abs/1312.6114) papers.
