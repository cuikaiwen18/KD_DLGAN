# KD-DLGAN: Data Limited Image Generation via Knowledge Distillation

### Updates


## Paper
![](./teaser.png)
[KD-DLGAN: Data Limited Image Generation via Knowledge Distillation](https://arxiv.org/abs/2303.17158)  
 Kaiwen Cui, Yingchen Yu, Fangneng Zhan, Shengcai Liao, Shijian Lu, Eric Xing 
 
 
 School of Computer Science Engineering, Nanyang Technological University, Singapore  
Computer Vision and Pattern Recognition Conference, 2023.


## Abstract
Generative Adversarial Networks (GANs) rely heavily on large-scale training data for training high-quality image generation models. With limited training data, the GAN discriminator often suffers from severe overfitting which directly leads to degraded generation especially in generation diversity. Inspired by the recent advances in knowledge distillation (KD), we propose KD-DLGAN, a knowledge-distillation based generation framework that introduces pre-trained vision-language models for training effective data-limited generation models. KD-DLGAN consists of two innovative designs. The first is aggregated generative KD that mitigates the discriminator overfitting by challenging the discriminator with harder learning tasks and distilling more generalizable knowledge from the pre-trained models. The second is correlated generative KD that improves the generation diversity by distilling and preserving the diverse image-text correlation within the pre-trained models. Extensive experiments over multiple benchmarks show that KD-DLGAN achieves superior image generation with limited training data. In addition, KD-DLGAN complements the state-of-the-art with consistent and substantial performance gains.

## DiffAugment for BigGAN
This repo is implemented upon the [BigGAN-PyTorch repo](https://github.com/ajbrock/BigGAN-PyTorch).

### Training
To train over CIFAR
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/DiffAugment-biggan-cifar10-0.1.sh
```


 ## Related Works
 We also would like to thank great works as follows:
 - https://github.com/NVlabs/stylegan2-ada-pytorch
 - https://github.com/mit-han-lab/data-efficient-gans


## Contact
If you have any questions, please contact: kaiwen001@e.ntu.edu.sg
