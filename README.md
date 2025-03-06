# Historical Document OCR System (layout_recognition part)

A comprehensive system for Optical Character Recognition (OCR) on historical documents with layout analysis capabilities.

## Project Overview

This project provides tools for processing historical documents through:
1. Layout recognition to identify text regions
2. Text extraction from identified regions
3. Post-processing of extracted text

The system uses deep learning models (U-Net, ResNet-UNet, and CNN-RNN hybrid architectures) to identify text regions in document images before performing OCR.

## Project Structure

```markdown

├── data/
│   ├── processed/
│   │   ├── annotations/              # JSON files with text region coordinates
│   │   └── images/                   # Source document images
│   └── prepared/                     # Processed images and masks for training
├── layout_recognition/
│   ├── data_preparation_script.py    # Prepares data for training
│   ├── evaluate_script.py            # Evaluates trained models
│   ├── model_script.py               # Model architecture definitions
│   ├── predict.py                    # Makes predictions on new documents
│   └── train_script.py               # Trains layout recognition models
├── models/
│   └── layout/                       # Saved model checkpoints
├── results/
│   ├── layout/                       # Evaluation results and visualizations
│   └── predictions/                  # Prediction results on new documents
└── scripts/
    ├── annotation_tool.py            # Tool for annotating text regions
    ├── pdf_to_images.py              # Conerts pdfs to images
    └── image_normalizer.py           # Changes resolution of the raw image to match annotations
 ```

## Model Architectures
### U-Net
A standard U-Net architecture for semantic segmentation with encoder-decoder structure.

### ResNet-UNet
U-Net with ResNet34 backbone pre-trained on ImageNet for feature extraction.

### CNN-RNN Model
Hybrid architecture combining ResNet feature extraction with LSTM for sequential processing, designed to better capture text layout patterns.


## Usage
### Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

### Data Preparation
1. Place your document images in data/processed/images/{source_folder}/
2. Create annotations using the annotation tool:
```bash
python scripts/annotation_tool.py
 ```
3. Save annotations to data/processed/annotations/{source_folder}/
4. Prepare data for training:
```bash
python layout_recognition/data_preparation_script.py
 ```

### Training
Train a layout recognition model:
```bash
python layout_recognition/train_script.py
 ```

### Evaluation
Evaluate a trained model:

```bash
python layout_recognition/evaluate_script.py
 ```

### Prediction
Process a new PDF document:

```bash
python layout_recognition/predict.py path/to/document.pdf
 ```
