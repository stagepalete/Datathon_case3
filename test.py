import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model

loaded_model = load_model('model')

def use_model(img):
    # image_path = f'techosmotr/test/{image_name}'
    # img = Image.open(image_path)
    img = img.resize((160, 160))  # Изменение размера изображения

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0  # Масштабирование значений пикселей

    predictions = loaded_model.predict(img_array)

    predicted_class_index = np.argmax(predictions)

    class_names = ['0', '1', '1', '1', '1']
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name
# def use_model(img):
#     # image_path = f'techosmotr/test/{image_name}'
#     # img = Image.open(image_path)
#     img = img.resize((220, 220))  # Resize the image
#
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#
#     img_array = img_array / 255.0  # Scale pixel values
#
#     predictions = loaded_model.predict(img_array)
#
#     predicted_class_index = np.argmax(predictions)
#
#     if predicted_class_index > 0:
#         return 1
#     else:
#         return 0

def to_submission():
    image_folder = 'techosmotr/test'
    sample_submission_path = 'sample_submission.csv'
    image_files = [file for file in os.listdir(image_folder) if file.endswith('.jpeg')]
    results_df = pd.DataFrame(columns=['file_index', 'class'])
    for image_file in image_files:
        # Extract the file index (assuming the file names are in the format "index.jpeg")
        file_index = os.path.splitext(image_file)[0]

        # Load and preprocess the image (use your own preprocessing steps)
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # Process the image using your model (replace with your model code)
        predicted_class = use_model(image)  # Assuming your model returns a class

        # Append the results to the DataFrame
        new_row = {'file_index': file_index, 'class': predicted_class}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    results_df.to_csv(sample_submission_path, index=False)

to_submission()
