import tensorflow as tf

def create_model(seq_length):
    input_shape = (seq_length, 3)
    inputs = tf.keras.Input(input_shape)

    # GRU Layers
    x = tf.keras.layers.GRU(128, recurrent_dropout=0.1)(inputs)

    # Separate Dense Layers for Outputs
    pitch_dense = tf.keras.layers.Dense(512, activation='relu')(x)
    step_dense = tf.keras.layers.Dense(64, activation='relu')(x)
    duration_dense = tf.keras.layers.Dense(64, activation='relu')(x)

    # Output Layers
    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(pitch_dense),
        'step': tf.keras.layers.Dense(1, name='step')(step_dense),
        'duration': tf.keras.layers.Dense(1, name='duration')(duration_dense),
    }

    return tf.keras.Model(inputs, outputs)
