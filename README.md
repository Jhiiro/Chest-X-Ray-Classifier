# Chest X-Ray Pneumonia Detection

I built this to get more comfortable with medical image classification. The task is straightforward: given a chest X-ray, predict whether the patient has pneumonia or not.

## The Data

I used the chest X-ray dataset from Kaggle (`paultimothymooney/chest-xray-pneumonia`). It splits into train, val, and test folders, each with NORMAL and PNEUMONIA subfolders. The training set has around 5,200 images; the validation set is tiny — only 16 images — so validation accuracy during training is pretty noisy and I wouldn't read too much into it.

Images are resized to 150×150 and pixel values are scaled to [0, 1].

## Model

Nothing exotic here. A sequential CNN with three Conv2D + MaxPool blocks, then a Dense(256) layer, Dropout(0.5), and a single sigmoid output for binary classification.

Total trainable parameters: ~2.67M.

Training uses Adam and binary cross-entropy. I set it to run for 25 epochs, though I stopped the run early while experimenting — the saved outputs only show about 9 completed epochs, with training accuracy climbing from around 85% to 96%.

Data augmentation (rotation, zoom, horizontal flips, shear) is applied during training only.

## Setup
```bash
pip install tensorflow numpy pandas matplotlib kagglehub

The notebook expects the dataset at `chest_xray/train`, `chest_xray/val`, and `chest_xray/test`. If you're running on Kaggle the paths are already set up; locally you'll need to download the dataset and adjust the paths at the top of the notebook.
