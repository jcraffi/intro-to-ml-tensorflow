import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from utils import predict
import json


def main(): # Example code to run: python predict.py ./test_images/cautleya_spicata.jpg flowers_model.keras --top_k 5 --category_names label_map.json
    parser = argparse.ArgumentParser(description='Predict flower class from an image along with the probability of that class')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('saved_model', type=str, help='Path to saved_model')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category to names JSON file')
    
    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.saved_model,custom_objects={'KerasLayer':hub.KerasLayer})

    # Run the model
    top_k_values, top_k_indices = predict(args.image_path, model, top_k=args.top_k)
  
    # Add class names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        with open('output.txt', 'w') as output_file:
            for i in range(args.top_k):
                label = class_names[str(top_k_indices[i])]
                probability = top_k_values[i] * 100  # Convert to percentage
                output_file.write(f'Label: {label}, Probability: {probability:.2f}%\n')
    else:
        with open('output.txt', 'w') as output_file:
            for i in range(args.top_k):
                label = top_k_indices[i]
                probability = top_k_values[i] * 100  # Convert to percentage
                output_file.write(f'Class ID: {label}, Probability: {probability:.2f}%\n')

if __name__ == '__main__':
    main()