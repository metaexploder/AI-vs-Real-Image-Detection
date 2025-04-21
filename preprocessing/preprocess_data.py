import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data = []
labels = []

# Path Setup
categories = ["real", "ai_generated"]
IMG_SIZE = 128

for category in categories:
    path = f"data/{category}"
    label = categories.index(category)  # 0 = real, 1 = ai_generated

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

# Convert to NumPy arrays
data = np.array(data) / 255.0  # Normalize pixels
labels = np.array(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Data Ready! Training samples: {len(X_train)}, Test samples: {len(X_test)}")
