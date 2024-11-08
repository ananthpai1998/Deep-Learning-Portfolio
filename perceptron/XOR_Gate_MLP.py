import multi_layer_perceptron
import gif_generator
import numpy as np

# Define a simple dataset (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

folder_to_save_frames = r'C:\Users\anant\Desktop\website repo\project_repo\Deep Learning Portfolio\perceptron\gifs'
folder_to_save_gif='C:/Users/anant/Desktop/website repo/project_repo/Deep Learning Portfolio/perceptron/gif_outputs/mlp_output_60.gif'

# Create and train the perceptron
mlp = multi_layer_perceptron.MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.4, epochs=5000, gif_folder_path=folder_to_save_frames)

mlp.train(X, y, animate=True)
print('Training complete!, creating gif...')
gif_generator.create_gif(frames_folder=folder_to_save_frames, output_folder=folder_to_save_gif, fps=5)
print('Gif created!')