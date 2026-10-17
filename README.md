# Cat and Dog Image Classification using HOG and SVM

## Overview

This project uses machine learning to classify images into two categories: cats and dogs.

Histogram of Oriented Gradients (HOG) is used to extract features from images, and a Support Vector Machine (SVM) classifier learns to distinguish between the two categories.

## Technologies Used

- Python
- NumPy
- Pillow
- Scikit-image
- Scikit-learn
- Matplotlib
- Joblib

## Dataset

The dataset used in this project is Microsoft's Cats vs Dogs dataset.

Download the dataset from:
https://www.microsoft.com/en-us/download/details.aspx?id=54765

After downloading, arrange the images in the following directory structure:

    dataset/
    └── PetImages/
        ├── Cat/
        └── Dog/

The dataset is not included in this repository.

The training script loads up to 500 images from each category and skips images that cannot be processed.

## Methodology

1. Load cat and dog images.
2. Convert images to RGB format.
3. Resize images to 64 × 64 pixels.
4. Extract image features using Histogram of Oriented Gradients (HOG).
5. Train a Support Vector Machine (SVM) classifier using the extracted features.
6. Evaluate the model using test accuracy and a confusion matrix.
7. Save the trained model locally for future predictions.

## Installation

Clone this repository and navigate to the project directory.

Install the required dependencies:

    pip install -r requirements.txt

## Training the Model

Run the training script:

    python src/model_train.py

The script trains the HOG + SVM classifier, evaluates its performance, generates a confusion matrix, and saves the trained model locally.

The dataset and trained model are excluded from this repository.

## Predicting a New Image

After training the model, run:

    python src/predict_image.py "path/to/your/image.jpg"

Replace the example path with the actual path of the image you want to classify.

The program predicts whether the image contains a cat or a dog.

## Results

- Classification algorithm: Support Vector Machine (SVM)
- Feature extraction: Histogram of Oriented Gradients (HOG)
- Image size: 64 × 64 pixels
- Maximum dataset sample: 1,000 images
- Test accuracy: 73.5%

The model was evaluated on a held-out test set of 200 images.

### Confusion Matrix

The confusion matrix visualizes the model's classification results for cats and dogs.

The model achieved 73.5% accuracy on the test set. Its performance may vary with different training samples and images.

## Limitations

- The model was trained on a limited sample of images.
- Classification performance may be affected by image quality, lighting, backgrounds, and variations in animal appearance.
- The difference between training and test accuracy suggests possible overfitting.

## Author

Developed as part of the SkillCraft Technology Machine Learning Internship.
