# -*- coding: utf-8 -*-
"""
Data Preparation and Latent Space Analysis for VAE MNIST Visualization
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers
from sklearn.neighbors import KNeighborsClassifier

# Configuration
MODEL_PATHS = {
    'encoder': '/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/encoder.keras',
    'decoder': '/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/decoder.keras'
}

DATA_CONFIG = {
    'num_classes': 10,
    'samples_per_class': 1000,
    'image_shape': (28, 28, 1)
}

# Custom Layers
@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Reparameterization trick layer for VAE."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=(ops.shape(z_mean)[0], ops.shape(z_mean)[1]))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

def load_models():
    """Load pre-trained VAE encoder and decoder models."""
    encoder = tf.keras.models.load_model(MODEL_PATHS['encoder'])
    decoder = tf.keras.models.load_model(MODEL_PATHS['decoder'])
    return encoder, decoder

def prepare_data():
    """Prepare MNIST data for latent space analysis."""
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255

    x_samples, y_samples = [], []
    for cls in range(DATA_CONFIG['num_classes']):
        indices = np.where(y_train == cls)[0]
        np.random.shuffle(indices)
        selected_indices = indices[:DATA_CONFIG['samples_per_class']]
        x_samples.append(x_train[selected_indices])
        y_samples.append(y_train[selected_indices])

    return np.concatenate(x_samples, axis=0), np.concatenate(y_samples, axis=0)

def generate_latent_space(encoder, x_samples):
    """Generate latent space representations using the encoder."""
    z_mean, _, _ = encoder.predict(x_samples, verbose=0)
    return z_mean

def train_knn(z_mean, y_samples):
    """Train KNN classifier on the latent space."""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(z_mean, y_samples)
    return knn

def generate_grid_data(z_mean, knn):
    """Generate grid data for KNN visualization."""
    grid_x = np.linspace(min(z_mean[:, 0]) - 1, max(z_mean[:, 0]) + 1, 500)
    grid_y = np.linspace(min(z_mean[:, 1]) - 1, max(z_mean[:, 1]) + 1, 500)
    grid = np.array(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T
    grid_labels = knn.predict(grid)
    return grid, grid_labels

def calculate_class_means(z_mean, y_samples):
    """Calculate mean latent space position for each class."""
    class_means = {}
    for cls in range(DATA_CONFIG['num_classes']):
        indices = np.where(y_samples == cls)[0]
        class_means[cls] = {
            'mean': np.mean(z_mean[indices], axis=0),  # Remove redundant np.array()
            'label': f'Class {cls}'
        }
    return class_means



if __name__ == "__main__":
    print("Preparing data for latent space analysis...")
    encoder, decoder = load_models()
    print("Models loaded successfully.")
    x_samples, y_samples = prepare_data()
    print("Data prepared successfully.")
    z_mean = generate_latent_space(encoder, x_samples)
    print("Latent space representations generated successfully.")
    knn = train_knn(z_mean, y_samples)
    print("KNN classifier trained successfully.")
    grid, grid_labels = generate_grid_data(z_mean, knn)
    print("Grid data generated successfully.")
    class_means = calculate_class_means(z_mean, y_samples)
    print("Class means calculated successfully.")

    # Save processed data for visualization
    np.savez('/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/latent_space_data.npz', 
             z_mean=z_mean, y_samples=y_samples, 
             grid=grid, grid_labels=grid_labels, 
             class_means=class_means)
    print("Data saved successfully.")