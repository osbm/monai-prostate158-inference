
# Steps to reproduce scores

## Install requirements

```
pip install -r requirements.txt
```

## Download the dataset
Download and unzip [this file](https://huggingface.co/datasets/osbm/prostate158/blob/main/data.zip).

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="osbm/prostate158", filename="data.zip", repo_type="dataset")
!unzip {path} -d .
```

## Download the model

Download the `anatomy.pt` file from this [zenodo repository](https://zenodo.org/record/6397057)

You can use my zenodo file downloader for this step:
```python
from datasets import load_dataset
load_dataset("osbm/zenodo", "7040585_anatomy.pt")
```

## Run the inference script

In this step, the produced masks will be saved to the `test` folder as `predicted.nii.gz` files for every case.

- This step takes about 5 minutes in cpu.

```
python inference.py
```

## Run score calculator script

```
python score.py
```
# Results

DICE scores:

| Case | Background | Inner Prostate | Outer Prostate |
| ---| ---| ---| --- |
|1 |0.99 |  0.792 | 0.81  |
| 2|0.993 | 0.897 | 0.703 |
|3 |0.992 | 0.917 | 0.757 |
| 4|0.994 | 0.838 | 0.725 |
|5 |0.994 | 0.899 | 0.771 |
| 6|0.995 | 0.871 | 0.77  |
|7 |0.995 | 0.892 | 0.761 |
| 8|0.996 | 0.87 |  0.78  |
|9 |0.996 | 0.847 | 0.722 |
| 10|0.988 | 0.841 | 0.693 |
|11 |0.996 | 0.839 | 0.825 |
| 12 |0.994 | 0.831 | 0.817 |
| 13 |0.993 | 0.823 | 0.744 |
| 14 |0.996 | 0.859 | 0.842 |
| 15 |0.995 | 0.889 | 0.767 |
| 16 |0.996 | 0.83 |  0.684 |
| 17 |0.995 | 0.864 | 0.779 |
| 18 |0.989 | 0.857 | 0.636 |
| 19 |0.995 | 0.915 | 0.636 |
| Mean | 0.99378947 | 0.86163158 | 0.74852632 |
