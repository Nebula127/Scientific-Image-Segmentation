import tensorflow as tf

def conv_block(x, filters, dropout=0.15):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x

def build_unet(input_shape=(256, 256, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32, dropout=0.1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64, dropout=0.15)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128, dropout=0.2)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256, dropout=0.25)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512, dropout=0.3)

    u4 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256, dropout=0.25)

    u3 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128, dropout=0.2)

    u2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64, dropout=0.15)

    u1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32, dropout=0.1)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(c8)

    return tf.keras.Model(inputs, outputs)