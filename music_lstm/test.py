import numpy as np
import tensorflow as tf
import pretty_midi
import yaml
from data import load_data, notes_to_midi
import pandas as pd

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load model
model = tf.keras.models.load_model("final_model1127_real.keras")

# Load test dataset
_, _, test_ds = load_data(
    config["data_path"], 
    config["sequence_length"], 
    config["batch_size"]
)

# Get one sample
for notes, _ in test_ds.take(1):
    input_notes = notes.numpy()
    break

# Ensure input shape is correct
if input_notes.ndim == 3:  # Remove batch dimension if present
    input_notes = input_notes[0]  # 첫 번째 배치를 선택하여 (50, 3) 형태로 만듦

# Generate notes
generated_notes = []
prev_start = 0

for _ in range(100):  # Generate 100 notes
    predictions = model.predict(np.expand_dims(input_notes, axis=0))  # Add batch dimension
    pitch = np.argmax(predictions['pitch'][0])
    step = max(0, predictions['step'][0][0])
    duration = max(0, predictions['duration'][0][0])
    start = prev_start + step
    end = start + duration
    generated_notes.append((pitch, step, duration, start, end))
    prev_start = start
    input_notes = np.roll(input_notes, shift=-1, axis=0)
    input_notes[-1] = [pitch, step, duration]

# Save as MIDI
generated_notes = pd.DataFrame(
    generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end']
)
out_pm = notes_to_midi(generated_notes, config["output_file"], instrument_name="Acoustic Grand Piano")
print(f"Generated MIDI saved to {config['output_file']}")
