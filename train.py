import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import build_unet
from data import split_by_filename
from losses import combined_loss, dice_coef, iou_coef

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled")

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 2
EPOCHS = 50
VAL_RATIO = 0.2

def load_from_paths(img_paths, mask_paths):
    imgs = []
    masks = []
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), desc="Loading data", total=len(img_paths)):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask, tf.float32) / 255.0
        
        imgs.append(img)
        masks.append(mask)
    
    return np.array(imgs), np.array(masks)

def make_dataset(imgs, masks, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(imgs))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    print("Starting U-Net training for brain tumor segmentation...")

    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = split_by_filename(val_ratio=VAL_RATIO)
    
    print("Loading training data...")
    train_imgs, train_masks = load_from_paths(train_img_paths, train_mask_paths)
    print("Loading validation data...")
    val_imgs, val_masks = load_from_paths(val_img_paths, val_mask_paths)
    
    train_ds = make_dataset(train_imgs, train_masks, shuffle=True)
    val_ds = make_dataset(val_imgs, val_masks, shuffle=False)
    
    model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[dice_coef, iou_coef]
    )
    
    os.makedirs("models", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/best_unet.keras", 
                                          save_best_only=True, 
                                          monitor="val_dice_coef", 
                                          mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", patience=10, 
                                        restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coef", factor=0.5, 
                                            patience=5, mode="max")
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed successfully. Best model saved to models/best_unet.keras")