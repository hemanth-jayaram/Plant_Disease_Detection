# Plant Disease Recognition System

This project implements a deep learning model based on ResNet architecture to classify plant diseases from leaf images.

## Dataset
The dataset used is from Kaggle: Plant Disease Recognition Dataset

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the 'data' directory

## Project Structure
- `data/`: Contains the dataset
- `src/`: Source code
- `models/`: Saved model weights
- `notebooks/`: Jupyter notebooks for experimentation

## Usage
1. Train the model:
```bash
python src/train.py
```

2. Make predictions:
```bash
python src/predict.py --image_path path/to/image.jpg
```

## Model Architecture
The project uses a ResNet-based architecture for plant disease classification.
