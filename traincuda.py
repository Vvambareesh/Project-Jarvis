import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LSTM, Dense
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
import numpy as np

# GPU Settings
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Parameters
SPECTROGRAM_DIR = "E:\\ai\\audiomodel\\wav\\spectrograms"  
TRAIN_DIR = "E:\\ai\\audiomodel\\wav\\train_dataset"
VAL_DIR = "E:\\ai\\audiomodel\\wav\\val_dataset"
TEST_DIR = "E:\\ai\\audiomodel\\wav\\test_dataset"

BATCH_SIZE = 32
NUM_SPECTROGRAM_BINS = 128
NUM_TIMESTEPS = 101

# Load training data
print("Loading training data...")
X_train = np.load(os.path.join(TRAIN_DIR, 'spectrograms.npy'))
y_train = np.load(os.path.join(TRAIN_DIR, 'transcripts.npy'))

# Load validation data 
print("Loading validation data...")
X_val = np.load(os.path.join(VAL_DIR, 'spectrograms.npy'))
y_val = np.load(os.path.join(VAL_DIR, 'transcripts.npy'))

with tf.device('/GPU:0'):

  # Model definition
  inputs = Input(shape=(NUM_SPECTROGRAM_BINS, NUM_TIMESTEPS, 1))

  x = Conv2D(64, 3, activation='relu')(inputs)
  x = Conv2D(128, 3, activation='relu')(x)
  x = Conv2D(256, 3, activation='relu')(x)

  x = LSTM(128, return_sequences=True)(x)
  x = LSTM(128)(x)

  x = Dense(256, activation='relu')(x)
  outputs = Dense(NUM_CHARS, activation='softmax')(x)

  # Compile and train model
  model = tf.keras.Model(inputs, outputs)
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  
  model.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, 
            epochs=10,
            validation_data=(X_val, y_val))
            
  # Save model
  model.save('my_tts_model_gpu.h5')
  
print("Saved model to disk")