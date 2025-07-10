# crack-detection-tensorflow-
A TensorFlow-based binary image classification project for detecting surface cracks in infrastructure using Convolutional Neural Networks (CNNs).
# Crack Detection using TensorFlow

A binary image classification project using Convolutional Neural Networks (CNNs) to detect cracks in images.

## Project Structure
- `data/`: Raw and preprocessed datasets
- `models/`: Saved trained models and checkpoints
- `notebooks/`: Jupyter notebooks (exploration, training)
- `scripts/`: Training & evaluation Python scripts
- `results/`: Accuracy plots, confusion matrix, etc.

##  Dataset
The dataset includes two classes:
- **Crack**: Images showing surface cracks
- **No Crack**: Images without cracks

Images are preprocessed and organized into a TensorFlow dataset using `image_dataset_from_directory`.

##  Model Architecture
- Input: 128x128 RGB images
- Convolutional layers with ReLU
- MaxPooling
- Flatten + Dense layers
- Output: Binary sigmoid activation

##  Evaluation Metrics
- Accuracy
- Loss curves
- Confusion matrix
- ROC AUC (optional)

##  Requirements
- Python 3.8+
- TensorFlow 2.x
- matplotlib, numpy, glob, etc.

## How to Run
```bash
# Train the model
python scripts/train_model.py

# Evaluate the model
python scripts/evaluate_model.py
