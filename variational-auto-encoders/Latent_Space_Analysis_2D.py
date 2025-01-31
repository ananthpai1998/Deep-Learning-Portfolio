# -*- coding: utf-8 -*-
"""Variational Autoencoder Latent Space Visualization Dashboard"""

# =============================================================================
# Imports & Model Setup
# =============================================================================
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
from dash import dcc, html, Input, Output, Dash

# =============================================================================
# Configuration Section
# =============================================================================
MODEL_PATHS = {
    'encoder': '/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/encoder.keras',
    'decoder': '/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/decoder.keras'
}

DATA_CONFIG = {
    'num_classes': 10,
    'samples_per_class': 1000,
    'image_shape': (28, 28, 1)
}

VISUALIZATION_CONFIG = {
    'contour_resolution': 100,
    'plot_height': 900,
    'colorscale': px.colors.qualitative.Plotly,
    'image_display_width': '300px'
}

APP_CONFIG = {
    'debug': True,
    'port': 8050
}



# =============================================================================
# Custom Layers (Required for model loading)
# =============================================================================
@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Reparameterization trick layer for sampling from latent space distribution."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

# =============================================================================
# Model Loading
# =============================================================================
vae_encoder = tf.keras.models.load_model(MODEL_PATHS['encoder'])
vae_decoder = tf.keras.models.load_model(MODEL_PATHS['decoder'])

# =============================================================================
# Data Preparation
# =============================================================================
# Load and preprocess MNIST data
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test, -1).astype("float32") / 255

# Create balanced class samples
sampled_images, sampled_labels = [], []
for cls in range(DATA_CONFIG['num_classes']):
    indices = np.where(y_test == cls)[0]
    np.random.shuffle(indices)
    selected_indices = indices[:DATA_CONFIG['samples_per_class']]
    sampled_images.append(x_test[selected_indices])
    sampled_labels.append(y_test[selected_indices])

sampled_images = np.concatenate(sampled_images, axis=0)
sampled_labels = np.concatenate(sampled_labels, axis=0)

# Generate latent space representations
z_mean, z_var, _ = vae_encoder.predict(sampled_images, verbose=0)

# =============================================================================
# Visualization Utilities
# =============================================================================
def plot_variance_contours(means, variances, colors, resolution=100):
    """
    Create variance contour plots for each class in latent space.
    
    Args:
        means: Dictionary of class means
        variances: Dictionary of class variances
        colors: Color scheme for contours
        resolution: Contour resolution
    
    Returns:
        List of plotly contour objects
    """
    contours = []
    for cls, mean in means.items():
        var_x, var_y = np.sqrt(np.abs(variances[cls][0])), np.sqrt(np.abs(variances[cls][1]))
        
        x = np.linspace(mean[0] - 3*var_x, mean[0] + 3*var_x, resolution)
        y = np.linspace(mean[1] - 3*var_y, mean[1] + 3*var_y, resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.exp(-((X - mean[0])**2/(2*np.abs(variances[cls][0])) + 
                     (Y - mean[1])**2/(2*np.abs(variances[cls][1]))))
        
        contours.append(go.Contour(
            x=x, y=y, z=Z,
            showscale=False,
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors[cls % len(colors)]]],
            opacity=0.2,
            line=dict(width=0),
            name=f'Class {cls} Variance'
        ))
    return contours

def generate_image_from_decoder(x, y):
    """Generate image from decoder given latent space coordinates."""
    latent_vector = np.array([[x, y]])
    generated_img = vae_decoder.predict(latent_vector, verbose=0)
    generated_img = (generated_img[0] * 255).astype(np.uint8).squeeze()
    return Image.fromarray(generated_img)

# =============================================================================
# Dashboard Setup
# =============================================================================
# Calculate class statistics
class_means = {cls: np.mean(z_mean[sampled_labels == cls], axis=0) 
               for cls in range(DATA_CONFIG['num_classes'])}
class_variances = {cls: np.mean(z_var[sampled_labels == cls], axis=0) 
                   for cls in range(DATA_CONFIG['num_classes'])}

# Create main visualization figure
mean_scatter = go.Scatter(
    x=[m[0] for m in class_means.values()],
    y=[m[1] for m in class_means.values()],
    mode='markers',
    marker=dict(
        size=10,
        color=VISUALIZATION_CONFIG['colorscale'],
        line=dict(color='black', width=1)
    ),
    name='Class Means'
)

fig = go.Figure(
    data=[mean_scatter] + plot_variance_contours(class_means, class_variances, 
                                               VISUALIZATION_CONFIG['colorscale']),
    layout=go.Layout(
        title='VAE Latent Space Visualization',
        xaxis_title='Latent Dimension 1',
        yaxis_title='Latent Dimension 2',
        showlegend=True,
        height=VISUALIZATION_CONFIG['plot_height']
    )
)

# Initialize Dash application
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='latent-space-plot', 
            figure=fig, 
            style={'flex': '2'}
        ),
        html.Img(
            id='generated-image', 
            style={
                'flex': '1',
                'width': VISUALIZATION_CONFIG['image_display_width'],
                'border': '2px solid black'
            }
        )
    ], style={'display': 'flex', 'gap': '20px', 'padding': '20px'})
])

# =============================================================================
# Application Callbacks
# =============================================================================
@app.callback(
    Output('generated-image', 'src'),
    Input('latent-space-plot', 'hoverData')
)
def update_image_on_hover(hover_data):
    """Update displayed image based on latent space hover position."""
    if hover_data:
        x, y = hover_data['points'][0]['x'], hover_data['points'][0]['y']
        img = generate_image_from_decoder(x, y)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    return ""

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    app.run_server(
        debug=APP_CONFIG['debug'], 
        port=APP_CONFIG['port']
    )
