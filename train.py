import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from model import create_model
from data import load_data

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load data
train_ds, val_ds, test_ds = load_data(
    config["data_path"], 
    config["sequence_length"], 
    config["batch_size"]
)

# Create model
model = create_model(config["sequence_length"])

# Compile model
loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'step': 'mse',
    'duration': 'mse',
}
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
    loss=loss
)

# Train model
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["epochs"],
    callbacks=callbacks
)

# Plot loss graph
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Save model
model.save("final_model.keras")
