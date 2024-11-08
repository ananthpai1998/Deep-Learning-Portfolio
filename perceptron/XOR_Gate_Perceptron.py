import perceptron
import gif_generator
import numpy as np

#Define a simple dataset (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

folder_to_save_frames = r'C:\Users\anant\Desktop\website repo\project_repo\Deep Learning Portfolio\perceptron\xor_perceptron_frames'
folder_to_save_gif='C:/Users/anant/Desktop/website repo/project_repo/Deep Learning Portfolio/perceptron/gif_outputs/xor_perceptron.gif'

# Create and train the perceptron
p = perceptron.Perceptron(input_size=2, learning_rate=0.1, epochs=50, gif_folder_path=folder_to_save_frames)
p.train(X, y)
print('Training complete!, creating gif...')
gif_generator.create_gif(frames_folder=folder_to_save_frames, output_folder=folder_to_save_gif, fps=5)
print('Gif created!')
