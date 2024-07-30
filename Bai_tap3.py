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
import numpy as np
import random

def create_triplets_batch(images, labels, batch_size):
    triplets = []
    for _ in range(batch_size):
        anchor_idx = random.randint(0, len(images) - 1)
        anchor_img = images[anchor_idx]
        anchor_label = labels[anchor_idx]

        positive_idx = random.choice(np.where(labels == anchor_label)[0])
        while positive_idx == anchor_idx:
            positive_idx = random.choice(np.where(labels == anchor_label)[0])
        positive_img = images[positive_idx]

        negative_idx = random.choice(np.where(labels != anchor_label)[0])
        negative_img = images[negative_idx]

        triplets.append((anchor_img, positive_img, negative_img))
    return np.array(triplets)

batch_size = 128
triplets_batch = create_triplets_batch(train_images, train_labels, batch_size)
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) - 1

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def relu(Z):
    return np.maximum(0, Z)

def forward_propagation(X, parameters):
    cache = {}
    A = X.T
    L = len(parameters) // 2

    for l in range(1, L + 1):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = relu(Z)
        cache['A' + str(l)] = A
        cache['Z' + str(l)] = Z

    return A.T, cache

# Example model configuration
layer_dims = [784, 128, 64, 32]
parameters = initialize_parameters(layer_dims)
A, cache = forward_propagation(triplets_batch[:, 0], parameters)
def compute_triplet_loss(A, P, N, alpha=0.2):
    pos_dist = np.sum(np.square(A - P), axis=1)
    neg_dist = np.sum(np.square(A - N), axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)
learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    triplets_batch = create_triplets_batch(train_images, train_labels, batch_size)
    A_anchor, _ = forward_propagation(triplets_batch[:, 0], parameters)
    A_positive, _ = forward_propagation(triplets_batch[:, 1], parameters)
    A_negative, _ = forward_propagation(triplets_batch[:, 2], parameters)

    loss = compute_triplet_loss(A_anchor, A_positive, A_negative)
    
    # Backpropagation and parameter updates would go here
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
