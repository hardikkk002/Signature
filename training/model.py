import tensorflow as tf


def build_encoder(input_shape=(224, 224, 1), embedding_size=128):
    inputs = tf.keras.Input(shape=input_shape, name="signature")
    x = inputs
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if filters != 256:
            x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(embedding_size)(x)
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        output_shape=(embedding_size,),
        name="embedding",
    )(x)
    return tf.keras.Model(inputs, outputs, name="signature_encoder")


def build_siamese(input_shape=(224, 224, 1), embedding_size=128):
    encoder = build_encoder(input_shape, embedding_size)
    a = tf.keras.Input(shape=input_shape, name="image_a")
    b = tf.keras.Input(shape=input_shape, name="image_b")
    ea, eb = encoder(a), encoder(b)
    distance = tf.keras.layers.Lambda(
        lambda pair: tf.sqrt(tf.reduce_sum(tf.square(pair[0] - pair[1]), axis=1, keepdims=True) + tf.keras.backend.epsilon()),
        output_shape=(1,),
        name="euclidean_distance",
    )([ea, eb])
    return tf.keras.Model([a, b], distance, name="siamese_signature_model")
