# -*- coding: utf-8 -*-
"""
Dash Application for VAE MNIST Latent Space Visualization
"""

import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
from dash import Dash, dcc, html, Input, Output

# Load pre-processed data
data = np.load('/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/latent_space_data.npz', allow_pickle=True)
z_mean, y_samples = data['z_mean'], data['y_samples']
grid, grid_labels = data['grid'], data['grid_labels']
class_means = data['class_means'].item()


# Load decoder model
decoder = tf.keras.models.load_model('/workspace/website repo/project_repo/Deep Learning Portfolio/variational-auto-encoders/decoder.keras')

def create_latent_space_plot():
    """Create the main latent space plot."""
    colors = px.colors.qualitative.Plotly
    
    # KNN classification background
    knn_colored_plot = go.Scatter(
        x=grid[:, 0], y=grid[:, 1], 
        mode='markers', 
        marker=dict(color=grid_labels, colorscale=colors, size=1),
        opacity=1,
        name='KNN Classification'
    )

    # Individual class means
    mean_scatter_list = []
    for cls, data in class_means.items():
        mean_scatter_list.append(go.Scatter(
            x=[data['mean'][0]],  # Direct array indexing
            y=[data['mean'][1]],
            mode='markers',
            marker=dict(
                size=10,
                color=colors[cls % len(colors)],
                line=dict(color='black', width=1)
            ),
            name=data['label']
        ))

    layout = go.Layout(
        title='VAE Latent Space with KNN Classification',
        xaxis_title='Latent Dimension 1',
        yaxis_title='Latent Dimension 2',
        showlegend=True,
        height=900
    )

    return go.Figure(data=[knn_colored_plot] + mean_scatter_list, layout=layout)


def generate_image_from_decoder(x, y):
    """Generate an image from the decoder given latent space coordinates."""
    z_input = np.array([[x, y]])
    generated_img = decoder.predict(z_input, verbose=0)
    generated_img = (generated_img[0] * 255).astype(np.uint8)
    img = Image.fromarray(generated_img.squeeze())
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='latent-space-plot', figure=create_latent_space_plot(), style={'flex': '2'}),
        html.Img(id='generated-image', style={'flex': '1', 'width': '300px', 'border': '2px solid black'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'})
])

@app.callback(
    Output('generated-image', 'src'),
    Input('latent-space-plot', 'hoverData')
)
def update_image_on_hover(hoverData):
    """Update the displayed image based on hover position in latent space."""
    if hoverData:
        x, y = hoverData['points'][0]['x'], hoverData['points'][0]['y']
        return f"data:image/png;base64,{generate_image_from_decoder(x, y)}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
