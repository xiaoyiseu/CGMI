## Usage
### Requirements
we use a single NVIDIA RTX 2080Ti  11G GPU for training and evaluation. 

```
python 3.8.19
torch 2.3.0+cu121
torchvision 0.18.0
numpy 1.22.1
pandas 2.0.3
pillow 10.3.0
```

## Training & Testing

python train.py
python test.py


## Result
Metric	accuracy	f1_macro	kappa	precision	sensitivity	specificity
Severity	0.8483	0.8477	0.6142	0.8511	0.8483	0.8511
Department	0.9089	0.9076	0.8587	0.9104	0.9089	0.9104