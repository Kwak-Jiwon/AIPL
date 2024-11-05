import tensorflow as tf
import numpy as np
import yaml
import matplotlib.pyplot as plt
from utils import load_data, create_sequences
from model import build_model

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 데이터 로드
data = load_data(config['data_path'])
key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([data[key] for key in key_order], axis=1)
notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
seq_ds = create_sequences(notes_ds, config['seq_length'], config['vocab_size'])

# 학습 및 검증 데이터셋 구성
train_size = int(0.8 * len(train_notes))  # 데이터의 80%를 학습 데이터로 사용
train_ds = (seq_ds.take(train_size)
            .shuffle(train_size)
            .batch(config['batch_size'], drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

val_ds = (seq_ds.skip(train_size)
          .batch(config['batch_size'], drop_remainder=True)
          .cache()
          .prefetch(tf.data.AUTOTUNE))

# 모델 생성
model = build_model((config['seq_length'], 3), config['learning_rate'])

# 모델 학습
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config['epochs']
)

# 학습 및 검증 손실 그래프 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
