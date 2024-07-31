import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
# Function to load and preprocess a custom image
def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Flatten the image to a 1D array (28*28,)
    image_array = image_array.flatten()
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    return image_array, image

# Path to your custom image
custom_image_path = 'image.png'

# Preprocess your custom image
custom_image, original_image = preprocess_image(custom_image_path)

# Predict the label of your custom image
custom_image_prediction = knn.predict([custom_image])
print("Predicted label for custom image:", custom_image_prediction[0])

# Display the custom image and the prediction
plt.imshow(original_image, cmap='gray')
plt.title(f"Predicted label: {custom_image_prediction[0]}")
plt.axis('off')
plt.show()
