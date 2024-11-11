import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path

def get_class_names(data_folder_path):
    class_names = [name for name in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, name))]
    class_names.sort()
    return class_names

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_binary_inv = 255 - image_binary

    final_image = cv2.resize(image_binary_inv, (75, 75))
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)
    final_image_rgb = final_image_rgb / 255.0
    final_image_rgb = final_image_rgb.reshape(1, 75, 75, 3)
    
    return final_image_rgb

def classify_image(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

model_path = os.path.join('.', 'Models', 'Pegon.ai.h5')
model = tf.keras.models.load_model(model_path)

class_names = get_class_names(os.path.join('.', 'Data'))

folder_path = Path(os.path.join('.', 'Test'))

for file in folder_path.iterdir():
    image_path = os.path.join('.', 'Test', file.name)
    predicted_class = classify_image(image_path)
    print(f'Prediksi : {class_names[predicted_class]} Asli : {file.name}', )