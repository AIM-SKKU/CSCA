# BL+CSCA for RGBT Crowd Counting 

We follow the official code of [Bayesian Loss for Crowd Count Estimation with Point Supervision (BL)](https://github.com/ZhihengCV/Bayesian-Crowd-Counting) and [Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting](https://github.com/chen-judge/RGBTCrowdCounting).

## Install dependencies
torch >= 1.0
torchvision
opencv
numpy
scipy
...

python 3.6

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

