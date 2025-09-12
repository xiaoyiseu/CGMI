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
```
python train.py
python test.py
```

## Result
| Metric| sensitivity | specificity |F1 score| kappa |
|:----:| ----:|----:|----:|----:|
| Severity| 0.8483 | 0.8511 | 0.8477 | 0.6142 |
| Department| 0.9089 | 0.9104 | 0.9076 | 0.8587 | 

## Citation
```
@InProceedings{XiaYi_ANovel_MICCAI2025,
        author = { Xiao, Yi and Zhang, Jun and Chi, Cheng and Wang, Chunyu},
        title = { { A Novel ED Triage Framework Using Conditional Imputation, Multi-Scale Semantic Learning, and Cross-Modal Fusion } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15969},
        month = {September},
        page = {13 -- 22}
}
```
