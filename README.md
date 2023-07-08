# DD-UGM
**Paper**: Universal Generative Modeling in Dual-domain for Dynamic MR Imaging

**Authors**: Chuanming Yu, Yu Guan, Ziwen Ke, Ke Lei, Dong Liang*, Qiegen Liu*

Date : June-13-2023  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Mathematics and Computer Sciences, Nanchang University. 

Dynamic magnetic resonance image reconstruction from incomplete k-space data has generated great research interest due to its capa-bility to reduce scan time. Nevertheless, the reconstruction problem remains a thorny issue due to its ill-posed nature. Recently, diffu-sion models, especially the score-based generative models, demonstrated great potential in terms of algorithmic robustness and flexi-bility of utilization. Moreover, the unified framework through the variance exploding stochastic differential equation (VE-SDE) is proposed to enable new sampling methods and further extend the capabilities of score-based generative models. Therefore, by taking advantage of the unified framework, we propose a k-space and image Dual-Domain collaborative Universal Generative Model (DD-UGM) which combines the score-based prior with low-rank regularization penalty to reconstruct highly under-sampled measurements. More precisely, we extract prior components from both image and k-space domains via a universal generative model and adaptively handle these prior components for faster processing while maintaining good generation quality. Experimental comparisons demon-strated the noise reduction and detail preservation abilities of the proposed method. Moreover, DD-UGM can reconstruct data of dif-ferent frames by only training a single frame image, which reflects the flexibility of the proposed model.

## Training Demo
``` bash
python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result
```

## Test Demo
``` bash
python PCsampling_demo_svd.py
```
## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Google Drive] (https://drive.google.com/file/d/1DmRTPmc_xYaVO3pX1R_CE0ZpiBRFkCwG/view?usp=sharing)

## Graphical representation
### Pipeline of the prior learning process and PI reconstruction procedure in WKGM
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-1.png" >  </div>
Top line: Prior learning is conducted in weight-k-space domain at a single coil. Bottom line: PI reconstruction is conducted in iterative scheme that alternates between WKGM update and other traditional iterative methods.

### Illustration of the forward and reverse processes of k-space data.
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-2.png" >  </div>

###  K-space domain and weight-k-space domain.
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-3.png" >  </div>
(a) The reference k-space data and its amplitude values. (b) The weight-k-space data and its amplitude values. (c) The image obtained by applying the inverse Fourier
encoding on k-space data. (d) The image obtained by applying the inverse Fourier encoding on weight-k-space data.

### PI reconstruction results
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-4.png" >  </div>
PI reconstruction results by ESPIRiT, LINDBERG, EBMRec, SAKE, WKGM and SVD-WKGM on T2 Transversal Brain image at R=10 using 2D Poisson sampling mask. The intensity of residual maps is five times magnify.

### 
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-5.png" >  </div>
Convergence curves of WKGM and SVD-WKGM in terms of PSNR versus the iteration number when reconstructing the brain image from 1/3 sampled data under 2D Poisson sampling pattern.

## Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.

## Other Related Projects    
* Low-rank Tensor Assisted K-space Generative Model for Parallel Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/2212/2212.05503.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/LR-KGM)     

* Universal Generative Modeling in Dual-domain for Dynamic MR Imaging  
[<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/2212/2212.07599.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DD-UGM) 

* One-shot Generative Prior in Hankel-k-space for Parallel Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2208.07181)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HKGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)

* Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
   
