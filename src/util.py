import pandas as pd
import os
import tensorflow as tf

def load_data(img_size = (224,224), batch_size = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/',
        validation_split=0.2,
        subset="training",
        seed=30,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/',
        validation_split=0.2,
        subset="validation",
        seed=30,
        image_size=img_size,
        batch_size=batch_size,
    )
    
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, train_ds.class_names
    