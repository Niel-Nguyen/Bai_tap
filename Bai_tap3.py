import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from PIL import Image

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

def compute_triplet_loss(A, P, N, alpha=0.2):
    pos_dist = np.sum(np.square(A - P), axis=1)
    neg_dist = np.sum(np.square(A - N), axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

def backward_propagation(X, A, P, N, parameters, cache, alpha):
    grads = {}
    L = len(parameters) // 2

    m = X.shape[0]
    pos_dist = A - P
    neg_dist = A - N
    dloss_dA = 2 * (pos_dist - neg_dist) / m

    for l in reversed(range(1, L + 1)):
        dZ = dloss_dA * relu(cache['Z' + str(l)].T)
        A_prev = cache['A' + str(l-1)].T if l > 1 else X

        grads['dW' + str(l)] = np.dot(dZ.T, A_prev) / m
        grads['db' + str(l)] = np.sum(dZ, axis=0, keepdims=True).T / m

        if l > 1:
            dloss_dA = np.dot(dZ, parameters['W' + str(l)])
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    
    return parameters

# Example model configuration
layer_dims = [784, 128, 64, 10]
parameters = initialize_parameters(layer_dims)

# Training loop
learning_rate = 0.01
num_epochs = 100
alpha = 0.2
batch_size = 128

for epoch in range(num_epochs):
    triplets_batch = create_triplets_batch(train_images, train_labels, batch_size)
    A_anchor, cache_anchor = forward_propagation(triplets_batch[:, 0], parameters)
    A_positive, cache_positive = forward_propagation(triplets_batch[:, 1], parameters)
    A_negative, cache_negative = forward_propagation(triplets_batch[:, 2], parameters)

    loss = compute_triplet_loss(A_anchor, A_positive, A_negative, alpha)
    
    grads_anchor = backward_propagation(triplets_batch[:, 0], A_anchor, A_positive, A_negative, parameters, cache_anchor, alpha)
    
    parameters = update_parameters(parameters, grads_anchor, learning_rate)
    
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32)
    img = img.flatten() / 255.0
    return img

def compute_embeddings(images, parameters):
    embeddings = []
    for image in images:
        embedding, _ = forward_propagation(np.array([image]), parameters)
        embeddings.append(embedding.flatten())
    return np.array(embeddings)

def predict_class(image, train_embeddings, train_labels, parameters):
    image_embedding, _ = forward_propagation(np.array([image]), parameters)
    image_embedding = image_embedding.flatten()

    # Use Nearest Neighbors to find the closest embedding
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train_embeddings)
    distances, indices = nbrs.kneighbors([image_embedding])
    
    # Get the label of the closest embedding
    predicted_label = train_labels[indices[0][0]]
    
    return predicted_label

# Compute embeddings for the training images
train_embeddings = compute_embeddings(train_images, parameters)

# Preprocess the external image
test_image = preprocess_image('7.png')

# Predict the class of the external image
predicted_label = predict_class(test_image, train_embeddings, train_labels, parameters)

print(f'Predicted label for the test image: {predicted_label}')
