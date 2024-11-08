import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from PIL import Image

def create_gif(frames_folder, output_folder, fps=50, frames_to_use=None):
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    if frames_to_use is None:
        frames_to_use = list(range(len(frame_files)))
    else:
        frames_to_use = [i for i in frames_to_use if 0 <= i < len(frame_files)]

    
    fig, ax = plt.subplots()
    def update(frame_index):
        print(frame_index)
        print(frames_to_use[frame_index])
        frame = frame_files[frames_to_use[frame_index]]
        img = Image.open(os.path.join(frames_folder, frame))
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
    anim = animation.FuncAnimation(fig, update, frames=len(frames_to_use), interval=10)
    anim.save(output_folder, writer='pillow', fps=fps)
    plt.close(fig)


# folder_to_save_frames = r'C:\Users\anant\Desktop\website repo\project_repo\Deep Learning Portfolio\perceptron\gifs'
# folder_to_save_gif='C:/Users/anant/Desktop/website repo/project_repo/Deep Learning Portfolio/perceptron/gif_outputs/mlp_output_60.gif'

# create_gif(frames_folder=folder_to_save_frames, output_folder=folder_to_save_gif, fps=50, frames_to_use=range(0, 1000, 1))
