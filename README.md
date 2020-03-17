# ReZero ResNet Unofficial Pytorch Implementation.

Trained a couple of nets for (fun) comparison, using identical hyperparams and early stopping on validation accuracy plateau schedule. All experiments can be replicated using the code from this repo.

Check out the **ReZero Paper** by the authors: https://arxiv.org/pdf/2003.04887.pdf \
Neat idea which seems to improve ResNet convergence speed, especially at the beggining of training.

## ReZero ResNet vs. ResNet on CIFAR-10:

| Model     | # params | runtime | epochs | Valid error (%) | Test error (%) |
|:-----------|:--------:|:--------:|:--------:|:-----------------:|:---------------------:|
| ResNet-20 | 272,474 | 70m3s | 398 |7.63 | **7.98** |
| ResNet-56 | 855,770 | 127m41s | 281 |6.04 | **6.44** |
| **ReZero** ResNet-20 | 272,483 | 63m9s | 327 |7.44 | **7.94** |
| **ReZero** ResNet-56 | 855,797 | 134m44s | 303 |6.31 | **6.55** |

## Loss & Error curves:
**ResNet-20:**

<img src="plots/resnet20_error.png" width="25%" height="25%"><img src="plots/resnet20_loss.png" width="25%" height="25%"><img src="plots/resnet20_error_0_30.png" width="25%" height="25%"><img src="plots/resnet20_loss_0_30.png" width="25%" height="25%">

**ResNet-56:**

<img src="plots/resnet56_error.png" width="25%" height="25%"><img src="plots/resnet56_loss.png" width="25%" height="25%"><img src="plots/resnet56_error_0_30.png" width="25%" height="25%"><img src="plots/resnet56_loss_0_30.png" width="25%" height="25%">

**This repo vs. ResNet paper:**
| Model     | (paper) Test error (%) | (this repo) Test error (%) |
|:-----------|:-----------------:|:---------------------:|
| ResNet-20 | 8.75 | **7.98** |
| ResNet-56 | 6.97 | **6.44** |
