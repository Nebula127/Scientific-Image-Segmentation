import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 256

def load_model_for_inference(model_path="models/best_unet.keras"):
    return tf.keras.models.load_model(model_path, compile=False)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def predict_mask(model, image_path, threshold=0.5):
    x = preprocess_image(image_path)
    pred = model.predict(x, verbose=0)[0, :, :, 0]
    pred_bin = (pred >= threshold).astype(np.uint8)
    return pred, pred_bin