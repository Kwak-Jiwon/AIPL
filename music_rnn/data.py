import os
import numpy as np
import pretty_midi

def load_midi_files(data_path, sequence_length):
    midi_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.mid', '.midi'))]
    print(f"Found MIDI files: {midi_files}")
    
    midi_data_list = []

    for midi_file in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            notes = midi_data.instruments[0].notes
            durations = []
            tempos = []
            pitches = []

            for i in range(0, len(notes) - sequence_length, sequence_length):
                sequence_notes = notes[i:i+sequence_length]
                sequence_duration = [note.end - note.start for note in sequence_notes]
                sequence_tempo = [midi_data.estimate_tempo()] * sequence_length
                sequence_pitch = [note.pitch for note in sequence_notes]

                durations.append(sequence_duration)
                tempos.append(sequence_tempo)
                pitches.append(sequence_pitch)

            midi_data_list.append({
                'durations': np.array(durations).reshape(-1, sequence_length, 1),
                'tempos': np.array(tempos).reshape(-1, sequence_length, 1),
                'pitches': np.array(pitches).reshape(-1, sequence_length, 1),
            })
                
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")

    print(f"Total MIDI files processed: {len(midi_data_list)}")
    return midi_data_list
