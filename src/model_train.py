import os
from PIL import Image
from skimage.feature import hog
import numpy as np

X = []
y = []

categories = {
    "Cat": 0,
    "Dog": 1
}
#convert images to HOG features and store them in X and y
for category, label in categories.items():
    image_path = f"dataset/PetImages/{category}"
    files = os.listdir(image_path)
    loaded = 0
    for file in files:
        if loaded >= 500:
            break

        try:
            img = Image.open(os.path.join(image_path, file))
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

            X.append(features)
            y.append(label)
            loaded += 1

        except Exception as e:
            print(f"Error occurred while processing {file}: {e}")

# Convert lists to numpy arrays

X = np.array(X)
y = np.array(y)



#Spilt the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#Train the SVM model(with RBF kernel)
from sklearn.svm import SVC
model = SVC(kernel="rbf", C=100, gamma="scale")
model.fit(X_train, y_train)
print("HOG + SVM model trained successfully!")

#predict on training and testing data and calculate accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("Training Accuracy:", train_accuracy * 100, "%")
print("Testing Accuracy:", test_accuracy * 100, "%")

#save the model using joblib
import joblib
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/hog_svm_model.pkl")
print("Model saved successfully!")

import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

display.plot()
plt.title("HOG + SVM Confusion Matrix")
plt.savefig("plots/hog_svm_confusion_matrix.png")
plt.show()