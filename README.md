<h1 align="center"> PI-Net </h1>

The official repository of **"PI-Net: A multi-site dataset for pixel-level segmentation and staging of pressure injuries."**

## Dataset Introduction
The dataset comprises two distinct subsets derived through different acquisition routes. 

The first subset, MIPI, consists of 406 original clinical wound photographs prospectively collected from six institutions, together with their corresponding pixel-level wound annotation masks and pressure injury (PI) stage labels. To reduce background interference, we additionally provide MIPI-C, a cropped version of MIPI that retains only the wound regions while preserving the original pixel-level wound annotations and PI stage labels. Both the MIPI images and their corresponding annotations, including the MIPI-C cropped images, are available for download from our Zenodo record: https://zenodo.org/records/22032032.

The second subset, PIID (Pressure Injury Image Dataset), contains 980 images re-curated from the publicly available PIID dataset originally released by Betul et al. The original 980 PIID images are not redistributed in our Zenodo record. Instead, they can be downloaded directly from the official FU-MedicalAI/PIID GitHub repository. For these 980 original images, we independently generated pixel-level wound segmentation masks and PI stage labels following the same standardized annotation protocol used for the MIPI subset. The newly generated PIID annotations are available for download from our Zenodo record: Zenodo record (MIPI and PIID annotations). All annotations were produced by certified wound care specialists.

## Data records
As shown in the figure, the root directory contains four main components: raw_data, nnunet_converted.py, metrics.ipynb, and README.md. The raw_data folder contains 406 original MIPI images and annotations, 406 MIPI-C images (cropped versions of the MIPI images) and their corresponding annotations, and 980 masks that we independently generated for the PIID images. Links to the original PIID images are also provided for download. The nnunet_converted.py script is used to convert images to the nnUNet-compliant format. The metrics.ipynb contains code to calculate the Dice coefficient and Intersection over Union (IoU) for evaluation. The README.md file provides a concise overview of the dataset structure and usage instructions.


## Download Papers and Datasets

**PI-Net Paper**
Link to be added   

**PI-Net Dataset**

* [Download MIPI images and annotations, MIPI-C, and PIID annotations from Zenodo](https://zenodo.org/records/22032032)
* [Download the original PIID images from the FU-MedicalAI/PIID repository](https://github.com/FU-MedicalAI/PIID)


## Dataset validation
### 1. train model
#### (1) Environment Variables Setup

##### Windows
Use the following commands (replace paths with your own):

```cmd
set nnUNet_raw=D:/DeepLearning/nnUNet/nnUNet_raw
set nnUNet_preprocessed=D:/DeepLearning/nnUNet/nnUNet_preprocessed
set nnUNet_results=D:/DeepLearning/nnUNet/nnUNet_results 
```

##### Linux
Use the following commands (replace paths with your own):
```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```

#### (2) preprocessed
```bash
nnUNetv2_plan_and_preprocess -d dataset-ID --verify_dataset_integrity
```

#### (3) train
This dataset only requires training 2D models
```bash
nnUNetv2_train dataset-ID 2d 0  
nnUNetv2_train dataset-ID 2d 1 
nnUNetv2_train dataset-ID 2d 2  
nnUNetv2_train dataset-ID 2d 3 
nnUNetv2_train dataset-ID 2d 4  
```

#### (4) Perform prediction on the test set
Use the following commands (replace paths with your own):
```bash
nnUNetv2_predict -d dataset -i nnUNet_raw\Dataset\imagesTs -o hippocampus_2d_predict -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities
```
### 2. performance evaluation
Dice-coefficient and IoU calculation:
```python
from pathlib import Path
from PIL import Image
import numpy as np

# Initialize lists to store results
dice_scores = []
iou_scores = []
file_names = []

# Original image path
images_dir = Path(r"text\imagesTr")        #Replace with your own path
# Predicted image path
pred_dir = Path(r"Dataset_ensemble_predict")

# Iterate through all images
for fn in images_dir.iterdir():
    try:
        # Original image path
        path_ori = fn
        # Predicted image path (using same filename without _0000)
        path_pred = pred_dir / path_ori.name.replace("_0000", "")
        # Ground truth label path (remove _0000)
        path_true = path_ori.parent.parent / "labelsTr" / path_ori.name.replace("_0000", "")

        print(f"Original image: {path_ori}")
        print(f"Predicted image: {path_pred}")
        print(f"Ground truth: {path_true}")

        # Open images
        image_ori = Image.open(path_ori)
        image_pred = Image.open(path_pred)
        image_true = Image.open(path_true)

        # Convert to NumPy arrays
        arr_pred = np.array(image_pred)
        arr_true = np.array(image_true)

        # Ensure inputs are binary
        arr_true = (arr_true > 0).astype(np.uint8)
        arr_pred = (arr_pred > 0).astype(np.uint8)

        # Ensure arrays have same shape
        if arr_pred.shape != arr_true.shape:
            print(f"Shape mismatch: {fn.stem} - Predicted:{arr_pred.shape}, Ground truth:{arr_true.shape}")
            continue

        # Calculate Dice and IoU
        intersection = np.sum(arr_true * arr_pred)
        union = np.sum(arr_true) + np.sum(arr_pred)

        # Avoid division by zero
        dice = (2.0 * np.sum(arr_true * arr_pred) + 1e-6) / (np.sum(arr_true) + np.sum(arr_pred) + 1e-6)
        iou = (np.sum(arr_true * arr_pred) + 1e-6) / (np.sum(arr_true + arr_pred - arr_true * arr_pred) + 1e-6)  ## Add smoothing term

        # Store results
        dice_scores.append(dice)
        iou_scores.append(iou)
        file_names.append(fn.stem)

        print(f"{fn.stem}: Dice={dice:.4f}, IoU={iou:.4f}")

    except Exception as e:
        print(f"Error processing {fn.stem}: {str(e)}")
        continue

# Calculate average scores (only if there's data)
if dice_scores:
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print("\nSummary results:")
    print(f"Average Dice coefficient: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Processed {len(dice_scores)} images in total")
else:
    print("\nWarning: No images were successfully processed! Please check paths and filenames.")
```

## License
The PI-Net dataset is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
