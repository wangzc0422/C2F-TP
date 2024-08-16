# C2F-TP: A Coarse-to-Fine Denoising Framework for Uncertainty-aware Trajectory Prediction (Under Review)
This repository contains the official implementation of C2F-TP: A Coarse-to-Fine Denoising Framework for Uncertainty-aware Trajectory Prediction.
<!--
## Background
Accurately predicting the trajectory of vehicles is critically important for ensuring safety and reliability in autonomous driving. Although considerable research efforts have been made recently, the inherent trajectory uncertainty caused by various factors including the dynamic driving intends and the diverse driving scenarios still poses significant challenges to accurate trajectory prediction. To address this issue, we propose C2F-TP, a coarse-to-fine denoising framework for uncertainty-aware vehicle trajectory prediction. C2F-TP features an innovative two-stage coarse-to-fine prediction process. Specifically, in the first stage we propose a spatial-temporal interaction module to capture the inter-vehicle interactions and learn a multimodal trajectory distribution, from which a certain number of noisy trajectories are sampled. In the trajectory refinement stage, we design a conditional denoising model to reduce the uncertainties of the sampled trajectories through a step-wise denoising operation. Extensive experiments are conducted on two real datasets NGSIM and highD that are widely adopted in trajectory prediction. The result demonstrates the effectiveness of our proposal. 

## Framework
-->
![image](https://github.com/wangzc0422/C2F-TP/blob/main/result/framework.png)
## Datasets
### NGSIM
The NGSIM dataset contains trajectories of real freeway traffic captured at 10 Hz over a time span of 45 minutes in 2015. It is collected on eastbound I-80 in the San Francisco Bay area and southbound US 101 in Los Angeles. Like the baselines, the NGSIM dataset in our work is segmented in the same way as in the most widely used work [Deo and Trivedi, 2018](https://github.com/nachiket92/conv-social-pooling), so that comparisons can be made.
### highD
The highD dataset consists of trajectories of 110000 vehicles recorded at 25 Hz, which are collected at a segment of two-way roads around Cologne in Germany from 2017 to 2018. Due to the policy requirements of this dataset, please request and download the HighD dataset from the [highD official website](https://www.highd-dataset.com/). Normally, applications take 7-14 working days to be approved.
## Running
### Train
We consider a two-stage training strategy to train C2F-TP as follows, which first trains the Refinement module and then trains the Spatio-Temporal Interaction module. <!--, where the first stage trains a denoising module and the second stage focuses on training a spatial-temporal interaction module. 
You can use the following command to start training C2F-TP.-->

<!-- - **Train the Refinement module.** -->
```
cd train
python train_denoise.py
python train_c2f.py
```
<!-- - **Freeze the parameters of the Refinement module and trains the Spatial-Temporal Interaction module.**
```
cd train
python train_c2f.py
```-->
### Evaluate
Run the following script to evaluate C2F-TP.
```
cd train
python evaluate_c2f.py
```
## Environment
Create a new Python environment (`c2f`) using `conda`:
```
conda create -n c2f python=3.7
conda activate c2f
```
Run the following script for environment configuration.
```
pip install -r requirements.txt
```
