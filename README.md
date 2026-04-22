# SPC-Gen: Synthetic PET and CT Data using Generative Model in Projection Domain

## Abstract
The rapid advancement of artificial intelligence has substantially increased the demand for large-scale datasets; however, available datasets remain insufficient for data-intensive tasks, particularly in nuclear medicine. While large scale datasets exist for computed tomography (CT), few provide in projection domain and image domain across diverse lesion types. This issue is more pronounced for positron emission tomography (PET), where datasets are scarce, particularly for multiple tracers such as ¹⁸F-DOPA, ¹⁸F-FDG, and ⁶⁸Ga-PSMA, anatomical regions, and disease categories. This work presents SPC-Gen, a large scale Synthetic dataset for PET and CT, Generated using a latent diffusion model. The dataset covers diverse disease types, lesion characteristics, tracers, anatomical regions, and modalities, while preserving the intrinsic correspondence between projection and image domains. SPC-Gen comprises over 50,000 paired projection and image slices for both PET and CT, forming a standardized large scale dataset. Experimental results show that the generated samples closely align with real data distributions and can support improved performance on downstream tasks such as disease classification. Compared with conventional VAE models and more recent GANs and NCSNs based baselines, the proposed approach maintains a stable KID of 0.021 and an FID of 40.960, indicating stronger and more consistent distribution alignment. 
  <tr>

## Model Architecture
The SPC-Gen generative framework is built on the Latent Diffusion Model (LDM) and VQ‑GAN to form an end-to-end projection-domain data generation pipeline, which first uses VQ‑GAN to map high-dimensional projection data into a compact discrete latent space, significantly reducing computational costs, and adopts a UNet architecture integrated with residual blocks and cross-attention layers as the core diffusion backbone; during the training process, the forward diffusion stage gradually injects Gaussian noise into real projection data, while the reverse diffusion stage relies on a denoising network to learn noise estimation and removal, enabling the generation of physically consistent projection data that can be directly reconstructed into clinical images via standard medical reconstruction pipelines. Furthermore, the entire model is trained with a low learning rate of 2.0×10⁻⁶ and a long diffusion schedule of 1000 steps to ensure stable convergence, high-fidelity data generation, and excellent reproducibility in large-scale dataset construction.
![](https://github.com/yqx7150/SPC-Gen/blob/main/Images%20uploaded%20to%20GitHub/fig1.png)

## Visual Results
1. Generated Samples vs. Baseline Methods
* Comparison between SPC-Gen and EnVAE, WGAN-QC, SGM in projection domain and image domain.
![](https://github.com/yqx7150/SPC-Gen/blob/main/Images%20uploaded%20to%20GitHub/fig2.png)
2. Multi-Tracer PET Synthetic Data
* Three-channel dual-tracer data covering brain, abdomen, and chest, including two single tracers and fused results.
![](https://github.com/yqx7150/SPC-Gen/blob/main/Images%20uploaded%20to%20GitHub/fig3.png)
3. PET Lesion Segmentation
* Trained with SPC-Gen synthetic data, the model achieves accurate lesion localization and delineation.
![](https://github.com/yqx7150/SPC-Gen/blob/main/Images%20uploaded%20to%20GitHub/fig5.png)
4. CT Lesion Segmentation
* Effectively identify and segment lung nodules, with high consistency with ground truth.
![](https://github.com/yqx7150/SPC-Gen/blob/main/Images%20uploaded%20to%20GitHub/fig6.png)
5. Low-Dose CT Denoising
* Suppresses noise and streaks under 50% low-dose conditions while preserving fine anatomical structures.
![](https://github.com/yqx7150/SPC-Gen/blob/main/Images%20uploaded%20to%20GitHub/fig7.png)

## Quantitative Evaluation
In terms of overall distribution alignment, SPC-Gen achieves a Kernel Inception Distance (KID) of 0.021, a lower value indicating stronger consistency with real data distributions; the Fréchet Inception Distance (FID) is 40.960; and the KL divergence for voxel intensity distribution is approximately 10⁻⁵, which fully demonstrates that the synthetic data closely matches the statistical distribution characteristics of real data in both the projection domain and image domain. For downstream task performance, models trained with SPC-Gen data achieve a Dice score of 82.42%, an IoU of 70.53%, and an average Hausdorff distance (aHD) of 6.93 mm for PET lesion segmentation; a Dice score of 79.34%, an IoU of 65.78%, and an aHD of 7.45 mm for CT lesion segmentation; and in low-dose CT denoising, SSIM is approximately 0.95 and PSNR is around 40 dB, showing excellent performance across all metrics. In instance-level validation using precision-recall based metrics, the Coverage Ratio (CR) remains stable at the theoretically optimal value of 0.50, the Nearest Neighbor Distance (NND) is low and stable, the Diversity Ratio (DR) is close to the ideal value of 1.0 without mode collapse, and the Composite Score (CS) is generally above 0.73 and can reach 0.79 to 0.80 in several dual-tracer and fusion settings, comprehensively validating the reliable performance of the synthetic data in structural fidelity, sample diversity, and generation stability.

## Quick Start
1. Clone Repository
  * git clone https://github.com/yqx7150/SPC-Gen.git
2. Configure Environment
  * conda env create -f environment.yaml
3. Download Dataset
  * Science Data Bank: https://doi.org/10.57760/sciencedb.3092948
  * GitHub Mirror: https://github.com/yqx7150/SPC-Gen
4. Run Inference & Generation
  * python main.py --config configs/latent-diffusion/unconditional/Single-channel.yaml
  * python main.py --config configs/latent-diffusion/unconditional/Multi-channel.yaml

## Application Scenarios
SPC-Gen can be directly used in a wide range of medical imaging and artificial intelligence research fields, including PET/CT image reconstruction and artifact correction, low-dose CT imaging and denoising, multi-tracer PET analysis and tracer translation, lesion detection, segmentation and classification, data augmentation for small-sample medical imaging tasks, the development of physically constrained artificial intelligence models, as well as quantitative nuclear medicine and radiomics research.

## Other Related Projects
* Raysolution_PET_Data [<font size=5>**[Data]**</font>](https://github.com/yqx7150/Raysolution_PET_Data)

* Diffusion Transformer Model with Compact Prior for Low-dose PET Reconstruction [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2407.00944)     [<font size=5>**[Code]**</font>](https://github.com/yqx7150/dtm)

* RED: Residual Estimation Diffusion for Low-Dose PET Sinogram Reconstruction  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1361841525001057)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/RED)
   
* Diffusion Transformer Meets Random Masks: An Advanced PET Reconstruction Framework [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2503.08339)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DREAM)

* Double-Constraint Diffusion Model with Nuclear Regularization for Ultra-low-dose PET Reconstruction  [<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2509.00395)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DCDM)
             
* Temporal Image Sequence Separation in Dual-tracer Dynamic PET with an Invertible Network  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10542421)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DTS-INN)             

* Synthetic CT Generation via Variant Invertible Network for Brain PET Attenuation Correction [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10666843) [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)

* Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10980366)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/st-DTPM)      
             
* PET Tracer Separation using Conditional Diffusion Transformer with Multi-latent Space Learning [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2506.16934#:~:text=In%20this%20study%2C%20a%20multi-latent%20space%20guided%20texture,model%20%28MS-CDT%29%20is%20proposed%20for%20PET%20tracer%20separation.)
      
* A Prior-Guided Joint Diffusion Model in Projection Domain for PET Tracer Conversion [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2506.16733) [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PJDM)    

* Positron Emission Tomography Tracer Conversion via Variable Augmented Invertible Network [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s12204-025-2844-2) 
