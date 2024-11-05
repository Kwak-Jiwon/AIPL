import tensorflow as tf
import numpy as np
import yaml
from utils import notes_to_midi
from model import build_model

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = build_model((config['seq_length'], 3), config['learning_rate'])
model.load_weights('./training_checkpoints/ckpt_final.weights.h5')

def predict_next_note(notes, model, temperature=1.0):
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch'] / temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    step = predictions['step']
    duration = predictions['duration']
    return int(pitch), float(step), float(duration)

# 시작 노트 설정
# ... (테스트 데이터 생성 및 음악 생성 코드 추가)
