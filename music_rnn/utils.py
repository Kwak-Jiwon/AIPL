import collections
import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
import pathlib
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def load_data(data_path, num_files=5):
    filenames = glob.glob(str(pathlib.Path(data_path) / '*.mid*'))
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)
    return pd.concat(all_notes)

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size: int = 128) -> tf.data.Dataset:
    seq_length = seq_length + 1
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        return x / [vocab_size, 1.0, 1.0]

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(['pitch', 'step', 'duration'])}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm
