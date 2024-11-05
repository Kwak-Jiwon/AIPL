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
