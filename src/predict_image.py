import os
import sys
import joblib
import numpy as np

from PIL import Image
from skimage.feature import hog

model = joblib.load("model/hog_svm_model.pkl")

image_path = sys.argv[1]
# Run using: python src/predict_image.py "image path"

img = Image.open(image_path)
img = img.convert("RGB")
img = img.resize((64, 64))

img_array = np.array(img)

features = hog(
    img_array,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    channel_axis=-1
)

features = features.reshape(1, -1)

prediction = model.predict(features)[0]

if prediction == 0:
    print("Prediction: Cat")
else:
    print("Prediction: Dog")