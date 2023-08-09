import os

import numpy as np
import pandas as pd
import torch
import nibabel as nib
import monai
from monai.metrics import DiceHelper

def DICE_COE(mask1, mask2, class_id):
    mask1 = mask1 == class_id
    mask2 = mask2 == class_id
    
    intersect = np.sum(mask1 * mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    
    dice = (2 * intersect) / (mask1_sum + mask2_sum)
    dice = round(dice, 3)  # for easy reading
    return dice

scores = []
for i in range(19):
    prediction = nib.load(f"all_to_all-outputs_post_processed/PROSTATE_{i+1}.nii.gz").get_fdata()
    ground_truth = nib.load(f"labelsTs/PROSTATE_{i+1}.nii.gz").get_fdata()

    for class_id in [0, 1, 2]:  # Loop over each class
        dice = DICE_COE(prediction, ground_truth, class_id)
        scores.append(dice)

scores = np.array(scores).reshape(19, 3)
df = pd.DataFrame(scores)
df.columns = ["background-dice", "inner-prostate-dice", "outer-prostate-dice"]
df.insert(0, "image-id", np.arange(1, 20))
df.to_csv("results.csv", index=False)

print("DICE scores:\n", scores)
print("Mean DICE scores:", np.mean(scores, axis=0))
