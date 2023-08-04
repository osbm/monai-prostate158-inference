import monai
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from monai.data import DataLoader
from monai.utils.enums import CommonKeys

from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    SaveImaged,
    ScaleIntensityd,
    NormalizeIntensityd,
    Spacingd,
    Orientationd,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = monai.networks.nets.UNet(
    in_channels=1,
    out_channels=3,
    spatial_dims=3,
    channels=[16, 32, 64, 128, 256, 512],
    strides=[2, 2, 2, 2, 2],
    num_res_units=4,
    act="PRELU",
    norm="BATCH",
    dropout=0.15,
)

model.load_state_dict(torch.load("anatomy.pt", map_location=device))

keys = ("t2", "t2_anatomy_reader1")
transforms = Compose(
    [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=[0.5, 0.5, 0.5], mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityd(keys=keys, minv=0, maxv=1),
        NormalizeIntensityd(keys=keys),
        EnsureTyped(keys=keys),
        ConcatItemsd(keys=("t2"), name=CommonKeys.IMAGE, dim=0),
        ConcatItemsd(keys=("t2_anatomy_reader1"), name=CommonKeys.LABEL, dim=0),
    ]
)

test_df = pd.read_csv("test.csv")
data_dict = [dict(row[1]) for row in test_df[list(keys)].iterrows()]


test_ds = Dataset(
    data=data_dict,
    transform=transforms,
)
model.eval()
with torch.no_grad():
    for i in range(len(test_ds)):
        inferer = monai.inferers.SlidingWindowInferer(
            roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5
        )
        input_tensor = test_ds[i]["t2"].unsqueeze(0)
        input_tensor = input_tensor.to(device)

        output_tensor = inferer(input_tensor, model)
        output_tensor = output_tensor.argmax(dim=1, keepdim=True)
        output_tensor = output_tensor.squeeze(0)
        output_tensor = output_tensor.to(torch.device("cpu"))
        output_tensor = output_tensor.numpy().astype(np.uint8)
        # save as nifti

        new_image = nib.Nifti1Image(output_tensor, affine=np.eye(4))
        nib.save(new_image, f"output/{i}.nii.gz")
        print(f"Saved {i}.nii.gz")
