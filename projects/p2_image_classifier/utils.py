from PIL import Image
import tensorflow as tf
import numpy as np

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k=5):
    # Preprocess the image
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    # Get predictions
    predictions = model.predict(np.expand_dims(processed_test_image, axis=0))
    # Get top K values and indices
    top_k_values, top_k_indices = tf.math.top_k(predictions, k=top_k)
    return top_k_values.numpy().tolist()[0], top_k_indices.numpy().tolist()[0]