import tensorflow as tf

def create_model(seq_length):
    input_shape = (seq_length, 3)
    inputs = tf.keras.Input(input_shape)

    # LSTM Layers
    x = tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1)(inputs)
    x = tf.keras.layers.LSTM(128, recurrent_dropout=0.1)(x)

    # L1 Regularization factor
    l1_factor = 0.01

    # Separate Dense Layers for Outputs with L1 Regularization
    pitch_dense = tf.keras.layers.Dense(
        1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_factor)
    )(x)
    step_dense = tf.keras.layers.Dense(
        64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_factor)
    )(x)
    duration_dense = tf.keras.layers.Dense(
        64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_factor)
    )(x)

    # Output Layers
    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(pitch_dense),
        'step': tf.keras.layers.Dense(1, name='step')(step_dense),
        'duration': tf.keras.layers.Dense(1, name='duration')(duration_dense),
    }

    return tf.keras.Model(inputs, outputs)

