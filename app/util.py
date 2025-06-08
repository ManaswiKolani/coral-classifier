import tensorflow
from tensorflow import keras
import numpy as np

def load_data(data_path: str, img_size=(256, 256), batch_size=32):
    data = keras.utils.image_dataset_from_directory(
        data_path,
        image_size=img_size,
        batch_size=batch_size
    )
    
    class_names = data.class_names

    data = data.map(lambda x, y: (x / 255.0, y))

    total_batches = len(data)
    train_size = int(total_batches * 0.7)
    val_size = int(total_batches * 0.2) + 1
    test_size = int(total_batches * 0.1) + 1

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    return train, val, test, class_names


def preprocess_image(image_path: str, target_size=(256, 256)):
    
    img = keras.utils.load_img(image_path, target_size=target_size)
    img_array = keras.utils.img_to_array(img) / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)

def load_trained_model(model_path: str):
    return keras.models.load_model(model_path)


def predict(model, processed_img, class_names: list):
    prediction = model.predict(processed_img)[0][0]
    class_idx = int(prediction >= 0.5)
    label = class_names[class_idx]
    confidence = prediction if class_idx == 1 else 1 - prediction
    return label, float(confidence)

