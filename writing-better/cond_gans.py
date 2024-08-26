import sys
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def main():
    try:
        # Read the input from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        # print(data)
        number = int(data['data'])
        testing_gen = load_model('/app/volume/model/cond_gans_generator_model.h5')

        one_hot_labels = tf.one_hot(tf.constant([int(number)]), 10)
        random_latent_vectors = tf.random.normal(shape=(1, 100))
        generated_images = testing_gen(tf.concat([random_latent_vectors, one_hot_labels], axis=-1))
        img_array = generated_images.numpy()[0]
        img_array = np.squeeze(img_array, axis=-1)
        img = Image.fromarray((img_array * 255).astype('uint8'))  # Convert to image format
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        
        # # Encode the image in Base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Print the image string (this will be sent as the response)
        print(json.dumps({"image": img_str}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        

if __name__ == "__main__":
    main()