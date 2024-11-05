import tensorflow as tf

def build_model(input_shape, learning_rate):
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1)(x)
    x = tf.keras.layers.LSTM(128, recurrent_dropout=0.1)(x)


    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': 'mse',
        'duration': 'mse',
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
