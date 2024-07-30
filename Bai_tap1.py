import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
def load_mnist(images_path, labels_path):
    with open(images_path, 'rb') as img_path:
        images = np.frombuffer(img_path.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with open(labels_path, 'rb') as lbl_path:
        labels = np.frombuffer(lbl_path.read(), np.uint8, offset=8)
    return images, labels

# Load training data
train_images, train_labels = load_mnist('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
# Load test data
test_images, test_labels = load_mnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_images, train_labels)

# Predict on the test set
test_predictions = knn.predict(test_images)

# Evaluate the model
accuracy = accuracy_score(test_labels, test_predictions)
print(accuracy)
