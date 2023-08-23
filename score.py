import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import monai
import numpy as np
import pandas as pd
import nibabel as nib
from keras.metrics import MeanIoU
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

n_classes = 3
IOU_keras = MeanIoU(num_classes=n_classes)  

scores = []
for i in range(19):
    # prediction = nib.load(f"all_to_all-outputs_post_processed/PROSTATE_{i+1}.nii.gz").get_fdata()
    # ground_truth = nib.load(f"labelsTs/PROSTATE_{i+1}.nii.gz").get_fdata()

    prediction = nib.load(f"test/{i+1:03}/predicted.nii.gz").get_fdata()
    ground_truth = nib.load(f"test/{i+1:03}/t2_anatomy_reader1.nii.gz").get_fdata()

    IOU_keras.update_state(ground_truth, prediction)
    for class_id in range(num_classes):  # Loop over each class
        scores.append(DICE_COE(prediction, ground_truth, class_id))

print("Mean IoU = ", IOU_keras.result().numpy())
print("Mean DICE scores:", np.mean(scores))
