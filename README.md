# Spatio-channel Attention Blocks for Cross-modal Crowd Counting (ACCV'22, Oral) -- Official Pytorch Implementation
Youjia Zhang, Soyun Choi, and Sungeun Hong."Spatio-channel Attention Blocks for Cross-modal Crowd Counting". The 16th Asian Conference on Computer Vision (ACCV), 2022.

Our proposed CSCA, a plug-and-play module, can achieve significant improvements for cross-modal crowd counting by simply integrating into various backbone network. You can refer to this code for implementing BL+CSCA for RGBT Crowd Counting. We follow the official code of [Bayesian Loss for Crowd Count Estimation with Point Supervision (BL)](https://github.com/ZhihengCV/Bayesian-Crowd-Counting) and [Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting](https://github.com/chen-judge/RGBTCrowdCounting).

## Install dependencies
torch >= 1.0
torchvision
opencv
numpy
scipy
...

python 3.6

## Method
The architecture of the proposed unified framework for extending existing baseline models from unimodal crowd counting to multimodal scenes. Our CSCA module is taken as the cross-modal solution to fully exploit the multimodal complementarities. Specifically, the CSCA consists of SCA to model global feature correlations among multimodal data, and CFA to dynamically aggregate complementary features.

![Architecture](https://github.com/zhangyj66/ACCV-2022-Spatio-channel-Attention-Blocks-for-Cross-modal-Crowd-Counting/blob/main/image/Architecture.jpg)

## Preprocessing

Edit the root and save path, and run this script:
```
python preprocess_RGBT.py
```


## Training
Edit this file for training BL-based CSCA model.
```
bash train.sh
```

## Testing
Edit this file for testing models.
```
bash test.sh
```

## Qualitative Results.
From the visualization results in cases (a) to (d) of the following figure, we can easily find that additional modality images can facilitate the crowd counting task better than only RGB images.  As we discussed earlier in the paper, inappropriate fusions fail to exploit the potential complementarity of multimodal data and even degrade the performance, such as the early fusion and late fusion shown in (e) and (f). Our proposed CSCA, a plug-and-play module, can achieve significant improvements for cross-modal crowd counting by simply integrating into the backbone network as shown in (g). This result shows the effectiveness of CSCA's complementary multi-modal fusion.
![Visualization](https://github.com/zhangyj66/ACCV-2022-Spatio-channel-Attention-Blocks-for-Cross-modal-Crowd-Counting/blob/main/image/Visualization.jpg)
