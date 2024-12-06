import tensorflow as tf
import pandas as pd
import numpy as np
import pretty_midi
import os
import glob
from typing import Tuple

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Converts a MIDI file into a DataFrame of notes."""
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]  # Assume single instrument
    notes = []
    
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start if sorted_notes else 0

    for note in sorted_notes:
        start = note.start
        end = note.end
        pitch = note.pitch
        notes.append({
            'pitch': pitch,
            'step': start - prev_start,
            'duration': end - start
        })
        prev_start = start

    return pd.DataFrame(notes)

def notes_to_midi(notes: pd.DataFrame, output_file: str, instrument_name: str = "Acoustic Grand Piano"):
    """Converts a DataFrame of notes into a MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for _, row in notes.iterrows():
        start = float(prev_start + row['step'])
        end = float(start + row['duration'])
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(row['pitch']),
            start=start,
            end=end
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(output_file)
    return pm

def create_sequences(dataset: tf.data.Dataset, seq_length: int) -> tf.data.Dataset:
    """Converts a dataset of notes into sequences and labels."""
    seq_length += 1  # Account for the label

    # Create windows of size seq_length
    windows = dataset.window(seq_length, shift=1, drop_remainder=True)
    sequences = windows.flat_map(lambda x: x.batch(seq_length))

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {'pitch': labels_dense[0], 'step': labels_dense[1], 'duration': labels_dense[2]}
        return inputs, labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def load_data(data_path: str, seq_length: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads MIDI data, preprocesses it into sequences, and splits into train/validation/test datasets.
    
    Args:
        data_path (str): Path to the folder containing MIDI files.
        seq_length (int): Length of each input sequence.
        batch_size (int): Batch size for training.
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Train, validation, and test datasets.
    """
    # Load all MIDI files
    files = glob.glob(os.path.join(data_path, "*.midi"))
    all_notes = []

    for file in files:
        notes = midi_to_notes(file)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes, ignore_index=True)

    # Convert notes to a numpy array
    notes_array = np.stack([all_notes['pitch'], all_notes['step'], all_notes['duration']], axis=1)

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(notes_array)

    # Create sequences and labels
    dataset = create_sequences(dataset, seq_length)

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Split into train, validation, and test datasets
    total_samples = len(files)
    train_size = int(0.7 * total_samples)
    val_size = int(0.2 * total_samples)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset
