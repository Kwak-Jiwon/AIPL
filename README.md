# AIPL: AI Music Composer and Lyric Synchronizer

## 👨‍🏫 Project Overview
This project aims to develop a neural network model for generating music using MIDI files. The system can analyze and compose new music based on MIDI data and also synchronize melodies with user-input lyrics through TTS (Text-to-Speech) technology. The primary technologies used in this project are TensorFlow-based RNN models and Tacotron for TTS integration.

The core of the system is built on MIDI analysis. MIDI files contain structured information about rhythm, harmony, and melody, which our model uses to learn musical patterns and generate new compositions. When lyrics are input, Tacotron generates a natural-sounding vocal melody, which is synchronized with the generated MIDI music to produce a seamless, complete musical experience.

## ⏲️ Development Period
- 2024-09-01 - present

## 🚀 Technologies Used
- **TensorFlow**: For building RNN models that analyze and generate MIDI sequences.
- **PrettyMIDI**: For reading and manipulating MIDI files.
- **Tacotron**: For generating speech synthesis that aligns with MIDI-generated music.
- **Python Libraries**: Including NumPy, Pandas, and Matplotlib for data processing and visualization.

## 🧑‍🤝‍🧑 Developer Information
- [Ji-eun Seo](https://github.com/Nick-Stokes)
- [Ji-won Kwak](https://github.com/Kwak-Jiwon)
- [Ji-min Kim](https://github.com/xxjimin)

---

## 📂 Project Structure

```plaintext
AIPL/
├── config.yaml                  # Configuration file for model parameters and data paths
├── train.py                     # Script to train the music generation model
├── generate.py                  # Script to generate new music using the trained model
├── model.py                     # Defines the LSTM model architecture
├── utils.py                     # Utility functions for data loading and MIDI processing
├── requirements.txt             # List of required libraries
└── README.md                    # Project documentation
```


## 🛠️ Usage

### Installation

Clone this repository and install the required dependencies:

```bash
git clone <repository_url>
cd AIPL
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to specify parameters like `data_path`, `seq_length`, and `learning_rate`. This configuration file should include:

```yaml
data_path: "path/to/your/midi/files"
seq_length: 25
vocab_size: 128
batch_size: 32
epochs: 10
learning_rate: 0.001
temperature: 2.0
num_predictions: 120
instrument_name: "Acoustic Grand Piano"
```

## Data Preparation

Place your MIDI files in the specified `data_path`. The `load_midi_files` function in `utils.py` will automatically load all `.mid` or `.midi` files from this directory.

## Training the Model

To train the model, run:

```bash
python train.py
```
This script will:

Load the configuration from config.yaml.
Prepare the MIDI data for training.
Build the RNN-based model.
Train the model and save the training history.
During training, the loss over epochs is plotted to help assess model performance.


## Generating Music

After training, use the following command to generate new music:

```bash
python generate.py
```

This will:

Load the trained model weights.
Generate new music sequences using the predict_next_note function.
Save the generated sequence as a MIDI file.



## Synchronizing Lyrics with Music
With Tacotron integrated, input lyrics can be synthesized into melody. This melody can then be aligned with the generated MIDI music, creating a fully harmonized audio file.

## 📈 Model Architecture
The model is built with TensorFlow and consists of:
- Multiple stacked LSTM layers for processing MIDI data.
- Dense output layers for predicting pitch, step, and duration of each note.


## 🖥️ Screenshots

## References
- PrettyMIDI Documentation
- TensorFlow
- Tacotron TTS

