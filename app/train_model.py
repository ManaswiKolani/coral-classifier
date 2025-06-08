import tensorflow as tf
from util import load_data
from model_architecture import build_model
import os

def main():
    DATA_PATH = 'clSData/'
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32
    EPOCHS = 10
    MODEL_PATH = 'Models/'
    LOGS_PATH = 'Logs/'

    # Load data
    train_ds, val_ds, test_ds, class_names = load_data(
        data_path=DATA_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Build model
    model = build_model(input_shape=(256, 256, 3))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOGS_PATH),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ]

    # Train model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save model
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_save_path = os.path.join(MODEL_PATH, 'model.h5')
    model.save(model_save_path)
    print(f"✅ Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
